import numpy as np
import pandas as pd
import sqlite3
from scipy.optimize import differential_evolution
import json
import time
import os
import math
from datetime import datetime
from colorama import Fore, Style, init
from numba import cuda, float64
import xgboost as xgb
import logging
from sklearn.utils.class_weight import compute_sample_weight
import warnings
from numba.core.errors import NumbaPerformanceWarning

# --- SILENCIAR LOGS DE NUMBA ---
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.cuda').setLevel(logging.WARNING)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
# -------------------------------

# --- CONFIGURACIÓN ---
DB_ETH = "cerebro_eth.db"
CONFIG_FILE = "config_params.json"
TIEMPO_ESPERA = 900

# Fee Binance Futures
FEE = 0.0005 

init(autoreset=True)

def cargar_datos(db_name, limite=900000):
    if not os.path.exists(db_name):
        print(f"{Fore.RED}[ERROR] No encuentro {db_name}")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_name)
        # Cargamos LIMIT para no saturar si hay millones, pero 900k es perfecto para tu GPU
        query = f"SELECT * FROM mercado_ticks ORDER BY timestamp DESC LIMIT {limite}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Ordenamos por timestamp ascendente (historia real)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Error SQL: {e}")
        return pd.DataFrame()

# --- KERNEL CUDA (ESTO CORRE DENTRO DE LA RTX 4070) ---
@cuda.jit
def gpu_backtest_kernel(bids, asks, imbs, ofis, ofi_ema_5s, ofi_ema_15s, prob_ups, prob_downs, ema_15m_dists, rsi_5ms, macro_sentiments, params, results):
    # Identificamos qué hilo (partícula) somos
    idx = cuda.grid(1)
    
    # Verificamos que no nos salgamos del array de partículas
    if idx < params.shape[0]:
        tp = params[idx, 0]
        sl = params[idx, 1]
        imb_thresh = params[idx, 2]
        conf_thresh = params[idx, 3]
        # Nuevos parametros para OFI
        ofi_thresh = params[idx, 4]
        ofi_ema_5_thresh = params[idx, 5]
        
        balance = 0.0
        posicion = 0 # 0: Nada, 1: Long, -1: Short
        entry_price = 0.0
        trades = 0
        wins = 0
        losses = 0
        
        # Constantes hardcoded para velocidad en GPU
        fee_rate = 0.0005 # Taker(0.05%) -> 0.10% total (0.05 * 2)
        leverage = 20.0
        
        ema_fast = 0.0
        ema_slow = 0.0
        alpha_fast = 2.0 / 3001.0
        alpha_slow = 2.0 / 36001.0
        prev_mid = (bids[0] + asks[0]) / 2.0
        
        n_rows = bids.shape[0]
        
        for i in range(n_rows):
            current_imb = imbs[i]
            bid = bids[i]
            ask = asks[i]
            mid = (bid + ask) / 2.0
            
            ret = abs(mid - prev_mid) / prev_mid
            prev_mid = mid
            
            if i == 0:
                ema_fast = ret
                ema_slow = ret
            else:
                ema_fast = alpha_fast * ret + (1.0 - alpha_fast) * ema_fast
                ema_slow = alpha_slow * ret + (1.0 - alpha_slow) * ema_slow
                
            is_normal_regime = True
            if i > 3000 and ema_slow > 0.0:
                ratio = ema_fast / ema_slow
                if ratio > 2.5 or ratio < 0.4:
                    is_normal_regime = False
            
            # 1. GESTIÓN DE SALIDA
            if posicion != 0:
                pnl_pct = 0.0
                closed = False
                
                if posicion == 1: # LONG
                    pnl_pct = (bid - entry_price) / entry_price # Sale cruzando el spread (vendiendo al Bid)
                    if pnl_pct >= tp or pnl_pct <= -sl:
                        closed = True
                        
                elif posicion == -1: # SHORT
                    pnl_pct = (entry_price - ask) / entry_price # Sale cruzando el spread (comprando al Ask)
                    if pnl_pct >= tp or pnl_pct <= -sl:
                        closed = True
                
                if closed:
                    # Cálculo de ganancia neta restando comisiones
                    gross_profit = pnl_pct * leverage
                    total_fees = (fee_rate * 2) * leverage
                    balance += (gross_profit - total_fees)
                    
                    if (gross_profit - total_fees) > 0:
                        wins += 1
                    else:
                        losses += 1
                        
                    posicion = 0
                    trades += 1
                    continue

            # 2. GESTIÓN DE ENTRADA
            if posicion == 0 and is_normal_regime:
                sentiment = macro_sentiments[i]
                tendencia_alcista = ema_15m_dists[i] >= -0.001 and rsi_5ms[i] < 70.0 and sentiment > -0.20
                tendencia_bajista = ema_15m_dists[i] <= 0.001 and rsi_5ms[i] > 30.0 and sentiment < 0.20
                
                if prob_ups[i] > conf_thresh and current_imb > imb_thresh and ofis[i] > ofi_thresh and ofi_ema_5s[i] > ofi_ema_5_thresh and tendencia_alcista:
                    posicion = 1
                    entry_price = ask # Orden Taker (cruza spread comprando al ask)
                elif prob_downs[i] > conf_thresh and current_imb < -imb_thresh and ofis[i] < -ofi_thresh and ofi_ema_5s[i] < -ofi_ema_5_thresh and tendencia_bajista:
                    posicion = -1
                    entry_price = bid # Orden Taker (cruza spread vendiendo al bid)
        
        # Guardamos resultado
        # Si opera muy poco (<5 trades), castigamos con costo alto
        if trades < 5:
            results[idx] = 1000000.0 
        else:
            win_rate = wins / trades
            # Exigimos matemáticamente un Win Rate realista de 65% (Muy rentable con R:R)
            if win_rate < 0.65:
                results[idx] = 500000.0 - balance # Castigo masivo
            else:
                results[idx] = -balance

def fitness_function_cuda(params, d_bids, d_asks, d_imbs, d_ofis, d_ofi_ema_5s, d_ofi_ema_15s, d_prob_ups, d_prob_downs, d_ema_15m_dists, d_rsi_5ms, d_macro_sentiments):
    n_particles = params.shape[0]
    
    # Reservamos memoria en GPU para los resultados de este lote
    results = np.zeros(n_particles, dtype=np.float64)
    d_results = cuda.to_device(results)
    
    # Copiamos los parámetros del enjambre a la GPU
    d_params = cuda.to_device(params)
    
    # Configuración de hilos
    threads_per_block = 128
    blocks_per_grid = (n_particles + (threads_per_block - 1)) // threads_per_block
    
    # LANZAMIENTO DEL KERNEL
    gpu_backtest_kernel[blocks_per_grid, threads_per_block](d_bids, d_asks, d_imbs, d_ofis, d_ofi_ema_5s, d_ofi_ema_15s, d_prob_ups, d_prob_downs, d_ema_15m_dists, d_rsi_5ms, d_macro_sentiments, d_params, d_results)
    
    # Esperamos a la GPU
    cuda.synchronize()
    
    # Traemos resultados a CPU
    return d_results.copy_to_host()

def entrenar_y_predecir_ia(df):
    model = xgb.XGBClassifier(
        n_estimators=80, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
        tree_method='hist', eval_metric='mlogloss',
        objective='multi:softprob', num_class=3, device='cuda', verbosity=0
    )
    
    # FIX ABSOLUTO: Asegurar columnas ANTES de cualquier cruce de Pandas
    for col in ['cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m']:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0)

    # Replicamos la logica de target del bot real (Time-Based Windowing)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # --- LOGICA DE TARGET REALISTA (CRUZANDO SPREAD) ---
    # La IA debe aprender a predecir si el precio futuro cruzara el spread actual
    df_target = df[['timestamp', 'best_bid', 'best_ask']].copy()
    df_target.rename(columns={'best_bid': 'future_bid', 'best_ask': 'future_ask'}, inplace=True)

    df['target_time'] = df['timestamp'] + 300.0 # 5 min para dar tiempo a que la tendencia se desarrolle
    
    merged = pd.merge_asof(
        df,
        df_target,
        left_on='target_time',
        right_on='timestamp',
        direction='forward'
    )
    
    merged = merged.dropna(subset=['future_bid', 'future_ask'])
    
    # Sincronizado a 0.25% para que la IA este alineada con los TP del bot
    MIN_PROFIT_PCT = 0.0025 # Retornado a 0.25% para optimizar movimientos más realistas de Taker
    
    # Para un LONG, entramos en 'best_ask'. El exito es salir en 'future_bid' por encima de nuestra entrada + profit.
    umbral_subida = merged['best_ask'] * (1 + MIN_PROFIT_PCT)
    # Para un SHORT, entramos en 'best_bid'. El exito es salir en 'future_ask' por debajo de nuestra entrada - profit.
    umbral_bajada = merged['best_bid'] * (1 - MIN_PROFIT_PCT)

    condiciones = [
        merged['future_bid'] > umbral_subida,
        merged['future_ask'] < umbral_bajada
    ]
    merged['target'] = np.select(condiciones, [1, 2], default=0)
    
    X = merged[['imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m']]
    y = merged['target']
    
    if len(y.unique()) < 2:
        print(f"{Fore.YELLOW}[WARNING] No hay suficientes clases unicas para entrenar la IA.")
        return None, None

    # --- FIX: Inyeccion de clases fantasma (Zero-Weight Dummies) ---
    missing_classes = list(set([0, 1, 2]) - set(y.unique()))
    
    if missing_classes:
        dummy_X = pd.DataFrame([X.iloc[-1]] * len(missing_classes), columns=X.columns)
        dummy_y = pd.Series(missing_classes)
        
        X_fit = pd.concat([X, dummy_X], ignore_index=True)
        y_fit = pd.concat([y, dummy_y], ignore_index=True)
    else:
        X_fit = X
        y_fit = y
        
    pesos = compute_sample_weight(class_weight='balanced', y=y_fit)
    
    # --- PENALIZACIÓN DE RUIDO ---
    # Obligamos al optimizador a valorar más las predicciones de movimientos fuertes
    pesos[y_fit.values == 0] *= 0.5
    
    if missing_classes:
        pesos[-len(missing_classes):] = 0.0
    
    model.fit(X_fit, y_fit, sample_weight=pesos)
    
    # Predecimos sobre el dataframe COMPLETO (con NaNs que luego se manejaran)
    X_infer = df[['imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m']].to_numpy(dtype=np.float32)
    
    # Inferencia Zero-Pandas ultrarrapida
    dmatrix = xgb.DMatrix(X_infer, feature_names=['imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m'])
    probs = model.get_booster().predict(dmatrix)
    return probs[:, 1], probs[:, 2]

def optimizar_moneda(simbolo, db_file):
    LIMITE_DATOS = 900000 
    print(f"\n{Fore.CYAN}[INFO] Optimizando {simbolo} con CUDA (RTX ACTIVADA)...")
    
    df = cargar_datos(db_file, limite=LIMITE_DATOS) 
    
    if df.empty or len(df) < 1000:
        print(f"{Fore.YELLOW}[WARNING] Pocos datos. Saltando.")
        return None

    # Cálculo de tiempo real (para ver proyección diaria)
    timestamp_inicial = df['timestamp'].iloc[0]
    timestamp_final = df['timestamp'].iloc[-1]
    dias_totales = (timestamp_final - timestamp_inicial) / 86400
    
    print(f"\n{Fore.CYAN}Data en VRAM: {dias_totales:.1f} dias ({len(df)} ticks)")

    # FIX ABSOLUTO: Asegurar columnas ANTES de cualquier cruce
    for col in ['cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'macro_sentiment']:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0)

    # --- CALCULO DE OFI (Order Flow Imbalance) y EMAs para el DF completo ---
    # Reconstrucción matemática de volúmenes individuales
    df['vol_bid'] = df['vol_total'] * (1 + df['imbalance']) / 2
    df['vol_ask'] = df['vol_total'] * (1 - df['imbalance']) / 2

    df['e_b'] = 0.0
    df['e_a'] = 0.0
    
    # Calculo de e_b
    df.loc[df['best_bid'] > df['best_bid'].shift(1), 'e_b'] = df['vol_bid']
    df.loc[df['best_bid'] == df['best_bid'].shift(1), 'e_b'] = df['vol_bid'] - df['vol_bid'].shift(1)
    df.loc[df['best_bid'] < df['best_bid'].shift(1), 'e_b'] = -df['vol_bid'].shift(1)
    
    # Calculo de e_a
    df.loc[df['best_ask'] < df['best_ask'].shift(1), 'e_a'] = df['vol_ask']
    df.loc[df['best_ask'] == df['best_ask'].shift(1), 'e_a'] = df['vol_ask'] - df['vol_ask'].shift(1)
    df.loc[df['best_ask'] > df['best_ask'].shift(1), 'e_a'] = -df['vol_ask'].shift(1)
    
    df['ofi'] = df['e_b'] - df['e_a']
    
    # EMAs iterativas (sin usar rolling de pandas para replicar la logica del bot)
    span_5 = 5
    alpha_5 = 2 / (span_5 + 1)
    span_15 = 15
    alpha_15 = 2 / (span_15 + 1)
    
    df['ofi_ema_5'] = 0.0
    df['ofi_ema_15'] = 0.0
    
    if len(df) > 0:
        df.loc[0, 'ofi_ema_5'] = df.loc[0, 'ofi']
        df.loc[0, 'ofi_ema_15'] = df.loc[0, 'ofi']
        for i in range(1, len(df)):
            df.loc[i, 'ofi_ema_5'] = alpha_5 * df.loc[i, 'ofi'] + (1 - alpha_5) * df.loc[i-1, 'ofi_ema_5']
            df.loc[i, 'ofi_ema_15'] = alpha_15 * df.loc[i, 'ofi'] + (1 - alpha_15) * df.loc[i-1, 'ofi_ema_15']

    df = df.dropna() # Eliminar NaNs generados por shift y EMAs iniciales

    # --- NUEVO: ENTRENAR IA Y GENERAR PREDICCIONES ---
    prob_ups, prob_downs = entrenar_y_predecir_ia(df.copy())
    if prob_ups is None:
        print(f"{Fore.RED}[ERROR] Fallo la simulacion de IA. Saltando optimizacion.")
        return None

    # --- CARGA MASIVA A VRAM ---
    print(f"\n{Fore.BLUE}[INFO] Transfiriendo arrays a GPU... Hecho.")
    # Convertimos a float64 explícito para Numba
    d_bids = cuda.to_device(df['best_bid'].to_numpy().astype(np.float64))
    d_asks = cuda.to_device(df['best_ask'].to_numpy().astype(np.float64))
    d_imbs = cuda.to_device(df['imbalance'].to_numpy().astype(np.float64))
    d_ofis = cuda.to_device(df['ofi'].to_numpy().astype(np.float64))
    d_ofi_ema_5s = cuda.to_device(df['ofi_ema_5'].to_numpy().astype(np.float64))
    d_ofi_ema_15s = cuda.to_device(df['ofi_ema_15'].to_numpy().astype(np.float64))
    d_prob_ups = cuda.to_device(prob_ups.astype(np.float64))
    d_prob_downs = cuda.to_device(prob_downs.astype(np.float64))
    d_ema_15m_dists = cuda.to_device(df['ema_15m_dist'].to_numpy().astype(np.float64))
    d_rsi_5ms = cuda.to_device(df['rsi_5m'].to_numpy().astype(np.float64))
    d_macro_sentiments = cuda.to_device(df['macro_sentiment'].to_numpy().astype(np.float64))
    # ---------------------------

    # --- LIMITES MEJORADOS (MAYOR WIN RATE Y MEJOR R:R) ---
    bounds = (
        (0.0020, 0.0060), # TP (0.20% a 0.60%) - Sincronizado con la IA
        (0.0050, 0.0120), # SL (0.50% a 1.20%) - Limitar el riesgo para no desangrar la cuenta
        (0.15, 0.40),
        (0.65, 0.95), # Mínimo 0.65 IA Conf para buscar más volumen de operaciones
        (0.05, 0.5),
        (0.05, 0.5)
    )
    
    result = differential_evolution(
        lambda p: fitness_function_cuda(np.ascontiguousarray(p.T), d_bids, d_asks, d_imbs, d_ofis, d_ofi_ema_5s, d_ofi_ema_15s, d_prob_ups, d_prob_downs, d_ema_15m_dists, d_rsi_5ms, d_macro_sentiments),
        bounds=bounds,
        strategy='best1bin',
        maxiter=50,
        popsize=200,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=False,
        vectorized=True,
        updating='deferred'
    )
    
    cost, pos = result.fun, result.x
    
    # Limpiamos VRAM
    d_bids = None
    d_asks = None
    d_imbs = None
    d_ofis = None
    d_ofi_ema_5s = None
    d_ofi_ema_15s = None
    d_prob_ups = None
    d_prob_downs = None
    d_ema_15m_dists = None
    d_rsi_5ms = None
    d_macro_sentiments = None
    
    if cost >= 400000: # Rechaza si el costo tiene la penalizacion de Win Rate (500k) o falta de trades (1M)
        print(f"{Fore.RED}[WARNING] {simbolo}: No rentable.")
        return None
        
    ganancia_proyectada = -cost
    ganancia_diaria = ganancia_proyectada / dias_totales if dias_totales > 0 else 0
    
    print(f"\n{Fore.GREEN}[SUCCESS] {simbolo} Optimizado (GPU).")
    print(f"[METRICA] Ganancia Total: ${ganancia_proyectada:.2f}")
    print(f"[METRICA] Proyeccion Diaria: ${ganancia_diaria:.2f} / dia")
    
    return {
        "take_profit": round(float(pos[0]), 5),
        "stop_loss": round(float(pos[1]), 5),
        "imbalance": round(float(pos[2]), 3),
        "ia_confidence": round(float(pos[3]), 3), # Confianza IA
        "ofi_threshold": round(float(pos[4]), 3), # Umbral OFI
        "ofi_ema_5_threshold": round(float(pos[5]), 3), # Umbral EMA 5
        "last_update": datetime.now().strftime("%H:%M:%S")
    }

def main():
    if not cuda.is_available():
        print(f"{Fore.RED}[ERROR CRITICO] GPU NO DETECTADA POR NUMBA.")
        print(f"{Fore.RED}[SOLUCION] Asegurese de ejecutar esto desde el entorno 'cerebro_gpu' de Conda.")
        return 
        
    print(f"{Fore.MAGENTA}=============================================")
    print(f"{Fore.MAGENTA}      CEREBRO V9 CUDA - POWERED BY NVIDIA    ")
    print(f"{Fore.MAGENTA}=============================================")
    
    try:
        gpu = cuda.get_current_device()
        print(f"Tarjeta Grafica: {gpu.name.decode('utf-8')}")
    except:
        print("Error obteniendo nombre de GPU, pero CUDA parece activo.")
    
    while True:
        try:
            current_config = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    try: current_config = json.load(f) # Cargar config actual
                    except json.JSONDecodeError: current_config = {} # Si el JSON esta vacio o corrupto
            
            # ETH
            res_eth = optimizar_moneda("ETH", DB_ETH)
            if res_eth: 
                current_config["ETH"] = res_eth

                # Guardar
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(current_config, f, indent=4) # Guardar manteniendo configuraciones de otras monedas
                print(f"\n{Fore.YELLOW}Configuracion actualizada.")
            
                # Reporte Visual
                p = current_config['ETH']
                print(f"ETH: TP {p['take_profit']*100:.2f}% | SL {p['stop_loss']*100:.2f}% | IMB {p['imbalance']} | IA_CONF {p.get('ia_confidence', 'N/A')} | OFI {p.get('ofi_threshold', 'N/A')} | OFI_EMA5 {p.get('ofi_ema_5_threshold', 'N/A')}")
            
            print(f"\n{Fore.BLUE}[INFO] Esperando 15 minutos...")
            time.sleep(TIEMPO_ESPERA)
            
        except KeyboardInterrupt:
            print("\n[SISTEMA] Apagando...")
            break
        except Exception as e:
            print(f"Error en loop principal: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()