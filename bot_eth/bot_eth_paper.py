import asyncio
import json
import websockets
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
import os
import sys
from colorama import Fore, Style, init
from datetime import datetime
import warnings
from sklearn.utils.class_weight import compute_sample_weight

# --- CONFIGURACION ---
SYMBOL_WSS = 'ethusdt'  
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(SCRIPT_DIR, "cerebro_eth.db")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "..", "config_params.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "paper_trading_log.txt")
STATE_FILE = os.path.join(SCRIPT_DIR, "sim_state_eth.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "modelo_ia_eth.json")

# PARAMETROS FIJOS
VENTANA_APRENDIZAJE = 8000
MONTO_USDT = 8.0  
LEVERAGE = 20
MAX_PERDIDA_DIARIA = -2.5 
COOLDOWN_SEGUNDOS = 300
HORA_INICIO_OP = 0   # 00:00 UTC
HORA_FIN_OP = 24     # 24:00 UTC (Todo el dia)

# COMISIONES BINANCE TAKER/TAKER (0.05% entrada + 0.05% salida) cruzando el spread
ROUND_TRIP_FEE = 0.0010 

# VARIABLES DINAMICAS
UMBRAL_CONFIANZA_IA = 0.85 
TAKE_PROFIT_PCT = 0.005
STOP_LOSS_PCT = 0.01
UMBRAL_IMBALANCE = 0.25
OFI_THRESHOLD = 0.01
OFI_EMA_5_THRESHOLD = 0.01

init(autoreset=True)
warnings.filterwarnings('ignore')

# --- FUNCIONES DE LOG Y CONFIG ---
def cargar_configuracion():
    global TAKE_PROFIT_PCT, STOP_LOSS_PCT, UMBRAL_IMBALANCE, UMBRAL_CONFIANZA_IA, OFI_THRESHOLD, OFI_EMA_5_THRESHOLD
    try:
        if not os.path.exists(CONFIG_FILE): 
            print(f"{Fore.YELLOW}[WARNING] No se encontro {CONFIG_FILE}. Creando configuracion por defecto...")
            default_config = {
                "ETH": {
                    "take_profit": TAKE_PROFIT_PCT,
                    "stop_loss": STOP_LOSS_PCT,
                    "imbalance": UMBRAL_IMBALANCE,
                    "ia_confidence": UMBRAL_CONFIANZA_IA,
                    "ofi_threshold": OFI_THRESHOLD,
                    "ofi_ema_5_threshold": OFI_EMA_5_THRESHOLD,
                    "last_update": "INICIAL"
                }
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)
            return True
            
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            params = data.get("ETH", {})
            if "take_profit" in params: TAKE_PROFIT_PCT = params["take_profit"]
            if "stop_loss" in params: STOP_LOSS_PCT = params["stop_loss"]
            if "imbalance" in params: UMBRAL_IMBALANCE = params["imbalance"]
            if "ia_confidence" in params: UMBRAL_CONFIANZA_IA = params["ia_confidence"]
            if "ofi_threshold" in params: OFI_THRESHOLD = params["ofi_threshold"]
            if "ofi_ema_5_threshold" in params: OFI_EMA_5_THRESHOLD = params["ofi_ema_5_threshold"]
            return True
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Error cargando configuracion: {e}")
        return False

def registrar_trade_log(tipo, entry_price, exit_price, pnl_pct, pnl_usd):
    try:
        hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resultado = "WIN" if pnl_usd > 0 else "LOSS"
        linea = f"[{hora}] {resultado} | {tipo} | IN: {entry_price:.2f} | OUT: {exit_price:.2f} | PNL%: {pnl_pct*100:.3f}% | NETO: ${pnl_usd:.4f}\n"
        with open(LOG_FILE, "a") as f:
            f.write(linea)
    except Exception as e:
        print(f"[ERROR] Error escribiendo log: {e}")

def guardar_estado_simulacion(posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido=6.0, rachas_perdidas=0, timestamp_entrada=0.0, confianza_ia=0.0):
    state = {
        "posicion": posicion,
        "precio_entrada": float(precio_entrada) if precio_entrada else 0.0,
        "max_pnl_pct": float(max_pnl_pct),
        "pnl_acumulado": float(pnl_acumulado),
        "trades_totales": int(trades_totales),
        "monto_invertido": float(monto_invertido),
        "rachas_perdidas": int(rachas_perdidas),
        "timestamp_entrada": float(timestamp_entrada),
        "confianza_ia": float(confianza_ia)
    }
    try:
        temp_file = STATE_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(state, f)
            
        # Reintento para evitar WinError 5 en Windows por bloqueo simultaneo de ADA
        for _ in range(10):
            try:
                os.replace(temp_file, STATE_FILE)
                break
            except PermissionError:
                time.sleep(0.01)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar estado simulado: {e}")

def cargar_estado_simulacion():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                st = json.load(f)
                return st.get("posicion"), st.get("precio_entrada", 0.0), st.get("max_pnl_pct", 0.0), st.get("pnl_acumulado", 0.0), st.get("trades_totales", 0), st.get("monto_invertido", 6.0), st.get("rachas_perdidas", 0), st.get("timestamp_entrada", 0.0), st.get("confianza_ia", 0.0)
        except: pass
    return None, 0.0, 0.0, 0.0, 0, 6.0, 0, 0.0, 0.0

# --- DETECTOR DE REGIMEN DE MERCADO ---
class MarketRegime:
    def __init__(self):
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.alpha_fast = 2.0 / (3000.0 + 1.0)  # ~5 mins
        self.alpha_slow = 2.0 / (36000.0 + 1.0) # ~1 hora
        self.prev_price = None
        self.ticks = 0
        self.current_regime = "CALIBRANDO"
        
    def update(self, price):
        if self.prev_price is None:
            self.prev_price = price
            return self.current_regime
            
        ret = abs(price - self.prev_price) / self.prev_price
        self.prev_price = price
        
        if self.ticks == 0:
            self.ema_fast = ret
            self.ema_slow = ret
        else:
            self.ema_fast = self.alpha_fast * ret + (1.0 - self.alpha_fast) * self.ema_fast
            self.ema_slow = self.alpha_slow * ret + (1.0 - self.alpha_slow) * self.ema_slow
            
        self.ticks += 1
        
        if self.ticks < 3000:
            return "CALIBRANDO"
            
        if self.ema_slow > 0:
            ratio = self.ema_fast / self.ema_slow
            if ratio > 2.5:
                self.current_regime = "SHOCK"
            elif ratio < 0.4:
                self.current_regime = "RANGO"
            else:
                self.current_regime = "NORMAL"
                
        return self.current_regime

# --- GESTOR DB ---
class GestorDB:
    """
    Gestor de base de datos SQLite optimizado con buffer en memoria
    para inserciones por lotes (batching), reduciendo el overhead de I/O.
    """
    def __init__(self, db_name, buffer_size=500):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_purge_time = 0
        self.crear_tabla()

    def crear_tabla(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mercado_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                mid_price REAL,
                imbalance REAL,
                spread REAL,
                wall_gap REAL,
                vol_total REAL,
                best_bid REAL,
                best_ask REAL,
                ofi REAL DEFAULT 0.0,
                ofi_ema_5 REAL DEFAULT 0.0,
                ofi_ema_15 REAL DEFAULT 0.0
            )
        ''')
        try:
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN cvd REAL DEFAULT 0.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN liq_longs REAL DEFAULT 0.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN liq_shorts REAL DEFAULT 0.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN ema_15m_dist REAL DEFAULT 0.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN rsi_5m REAL DEFAULT 50.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN btc_trend REAL DEFAULT 0.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN atr_5m REAL DEFAULT 0.0")
            self.cursor.execute("ALTER TABLE mercado_ticks ADD COLUMN macro_sentiment REAL DEFAULT 0.0")
        except:
            pass
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_time ON mercado_ticks(timestamp)')
        self.conn.commit()

    def guardar_tick(self, data):
        self.buffer.append((
            time.time(), 
            data['mid_price'], 
            data['imbalance'], 
            data['spread'], 
            data['wall_gap'], 
            data['vol_total'], 
            data['best_bid'], 
            data['best_ask'],
            data['ofi'],
            data['ofi_ema_5'],
            data['ofi_ema_15'],
            data['cvd'],
            data['liq_longs'],
            data['liq_shorts'],
            data['ema_15m_dist'],
            data['rsi_5m'],
            data['btc_trend'],
            data['atr_5m'],
            data['macro_sentiment']
        ))
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        """
        Ejecuta un volcado asincrono (batch insert) de los datos en memoria hacia SQLite.
        """
        if not self.buffer:
            return
            
        self.cursor.executemany('''
            INSERT INTO mercado_ticks (timestamp, mid_price, imbalance, spread, wall_gap, vol_total, best_bid, best_ask, ofi, ofi_ema_5, ofi_ema_15, cvd, liq_longs, liq_shorts, ema_15m_dist, rsi_5m, btc_trend, atr_5m, macro_sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', self.buffer) 
        self.conn.commit()
        self.buffer.clear()

    def obtener_datos_entrenamiento(self, limite=3000):
        self.flush()
        query = f"SELECT * FROM mercado_ticks ORDER BY timestamp DESC LIMIT {limite}"
        df = pd.read_sql_query(query, self.conn)
        if df.empty: return df
        return df.sort_values(by='timestamp')

    def purgar_datos_viejos(self, horas=12):
        """Borra los registros de la base de datos que son más antiguos que el número de horas especificado."""
        if time.time() - self.last_purge_time < 3600: # Cada hora
            return
        try:
            self.flush()
            limite_tiempo = time.time() - (horas * 3600)
            self.cursor.execute("DELETE FROM mercado_ticks WHERE timestamp < ?", (limite_tiempo,))
            self.conn.commit()
            
            if time.time() - getattr(self, 'last_vacuum_time', 0) > 86400: # Cada 24h
                 print(f"\n{Fore.BLUE}[DB] Ejecutando VACUUM para optimizar espacio...")
                 self.conn.execute("VACUUM")
                 self.conn.commit()
                 self.last_vacuum_time = time.time()

            self.last_purge_time = time.time()
        except Exception as e:
            print(f"\n{Fore.YELLOW}[DB WARNING] No se pudo purgar la base de datos: {e}")

    def close(self):
        self.flush()
        self.conn.close()

# --- DATOS MACRO (MULTI-TIMEFRAME) ---
def fetch_mtf_data(symbol):
    try:
        url_15m = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=20"
        res_15m = requests.get(url_15m, timeout=5).json()
        closes_15m = [float(k[4]) for k in res_15m]
        ema_15m = pd.Series(closes_15m).ewm(span=15, adjust=False).mean().iloc[-1]
        
        url_5m = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&limit=20"
        res_5m = requests.get(url_5m, timeout=5).json()
        df_5m = pd.DataFrame(res_5m, columns=['time','open','high','low','close','vol','close_time','qav','trades','taker_base','taker_quote','ignore'])
        df_5m = df_5m.astype(float)
        
        # Calculate RSI 5m
        delta = df_5m['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        if loss.iloc[-1] == 0: rsi_5m = 100.0
        else: rsi_5m = 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1])))
            
        # Calculate ATR 5m (14 periods)
        high_low = df_5m['high'] - df_5m['low']
        high_close = np.abs(df_5m['high'] - df_5m['close'].shift())
        low_close = np.abs(df_5m['low'] - df_5m['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr_5m = true_range.rolling(14).mean().iloc[-1]
        
        return ema_15m, float(rsi_5m), float(atr_5m)
    except Exception as e:
        return None, None, None

# --- CEREBRO IA ---
class CerebroIA:
    """
    Motor de Inferencia y Entrenamiento XGBoost con aceleracion hardware (CUDA).
    Implementa Zero-Pandas inference y Time-Based Windowing.
    """
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        self.model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=7,
            tree_method='hist', eval_metric='mlogloss',
            objective='multi:softprob', num_class=3, device='cuda',
            verbosity=0
        )
        self.trained = False
        self.last_train_time = 0
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                if not hasattr(self.model, 'classes_'):
                    self.model.classes_ = np.array([0, 1, 2])
                self.trained = True
                print(f"{Fore.GREEN}[IA] Modelo ETH cargado desde disco. Listo para operar.")
            except Exception as e:
                print(f"{Fore.YELLOW}[IA WARNING] Error cargando modelo previo: {e}")

    def entrenar(self, db):
        if time.time() - self.last_train_time < 1800: return None
        df = db.obtener_datos_entrenamiento(VENTANA_APRENDIZAJE)
        if len(df) < 200: return None
        
        # FIX: Retro-compatibilidad para evitar KeyError
        for col in ['cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'btc_trend', 'atr_5m', 'macro_sentiment']:
            if col not in df.columns: df[col] = 0.0
            else: df[col] = df[col].fillna(0.0)

        df = df.sort_values('timestamp').reset_index(drop=True)
        df_target = df[['timestamp', 'best_bid', 'best_ask']].copy()
        df_target.rename(columns={'best_bid': 'future_bid', 'best_ask': 'future_ask'}, inplace=True)
        
        df['target_time'] = df['timestamp'] + 300.0 # 5 minutos para evaluar la proyeccion futura
        
        merged = pd.merge_asof(
            df,
            df_target,
            left_on='target_time',
            right_on='timestamp',
            direction='forward'
        )
        
        merged = merged.dropna(subset=['future_bid', 'future_ask'])
        
        MIN_PROFIT_PCT = 0.0015 # Relajado a 0.15% para que coincida con el optimizador
        umbral_subida = merged['best_ask'] * (1 + MIN_PROFIT_PCT)
        umbral_bajada = merged['best_bid'] * (1 - MIN_PROFIT_PCT)
        
        condiciones = [
            merged['future_bid'] > umbral_subida,
            merged['future_ask'] < umbral_bajada
        ]
        merged['target'] = np.select(condiciones, [1, 2], default=0)
        
        X = merged[['imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'btc_trend', 'atr_5m', 'macro_sentiment']]
        y = merged['target']
        
        if len(y.unique()) < 2:
            return None
            
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
        if missing_classes:
            pesos[-len(missing_classes):] = 0.0
            
        try:
            self.model.fit(X_fit, y_fit, sample_weight=pesos)
            self.trained = True
            self.last_train_time = time.time()
            self.model.save_model(self.model_path)
            return self.model.score(X_fit, y_fit)
        except Exception as e:
            print(f"[ERROR] Falla en entrenamiento: {e}")
            return None

    def predecir(self, data_dict):
        """
        Inferencia en caliente Zero-Pandas. Procesa los diccionarios y
        construye una matriz Numpy float32 bidimensional para minima latencia.
        """
        if not self.trained: return 0.0, 0.0
        
        X = np.array([[
            data_dict['imbalance'], 
            data_dict['spread'], 
            data_dict['wall_gap'],
            data_dict['vol_total'], 
            data_dict['ofi'],
            data_dict['ofi_ema_5'],
            data_dict['ofi_ema_15'],
            data_dict['cvd'],
            data_dict['liq_longs'],
            data_dict['liq_shorts'],
            data_dict['ema_15m_dist'],
            data_dict['rsi_5m'],
            data_dict['btc_trend'],
            data_dict['atr_5m'],
            data_dict['macro_sentiment']
        ]], dtype=np.float32)
        
        dmatrix = xgb.DMatrix(X, feature_names=['imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'btc_trend', 'atr_5m', 'macro_sentiment'])
        probs = self.model.get_booster().predict(dmatrix)[0]
        
        prob_up = 0.0
        prob_down = 0.0
        for i, clase in enumerate(self.model.classes_):
            if clase == 1: prob_up = probs[i]
            elif clase == 2: prob_down = probs[i]
            
        return prob_up, prob_down

# --- MOTOR PRINCIPAL ---
async def main_loop(db, ia):
    print(f"{Fore.MAGENTA}=====================================================")
    print(f"{Fore.MAGENTA} [SISTEMA] ETHEREUM BOT - PAPER TRADING (SIMULACION)")
    print(f"{Fore.MAGENTA}=====================================================")
    
    # Variables de Flujo y MTF
    cvd_vol = 0.0
    cvd_ema = 0.0
    liq_longs = 0.0
    liq_shorts = 0.0
    ema_15m = None
    rsi_5m = 50.0
    last_mtf_update = 0

    cargar_configuracion()
    print(f"[CONFIG] IA {UMBRAL_CONFIANZA_IA} | IMB {UMBRAL_IMBALANCE} | TP {TAKE_PROFIT_PCT*100:.2f}% | SL {STOP_LOSS_PCT*100:.2f}% | OFI {OFI_THRESHOLD} | EMA5 {OFI_EMA_5_THRESHOLD}")
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write(f"--- INICIO DE PAPER TRADING LOG ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
    print(f"[LOGS] Guardando en: {LOG_FILE}")

    # --- VARIABLES DE ESTADO PARA OFI Y EMA ---
    prev_best_bid = None
    prev_vol_bid = None
    prev_best_ask = None
    prev_vol_ask = None
    prev_ofi_ema_5 = 0.0
    prev_ofi_ema_15 = 0.0

    # EMA spans
    span_5 = 5
    alpha_5 = 2 / (span_5 + 1)
    span_15 = 15
    alpha_15 = 2 / (span_15 + 1)
    
    regime_detector = MarketRegime()
    posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, confianza_ia = cargar_estado_simulacion()
    cooldown_actual = rachas_perdidas * COOLDOWN_SEGUNDOS if rachas_perdidas > 0 else 0

    confirmaciones_long = 0
    confirmaciones_short = 0
    TICKS_CONFIRMACION = 4 # 300ms: Filtra el ruido pero no asfixia la entrada
    confirmaciones_reversal = 0
    SPREAD_MAXIMO_PCT = 0.0008
    
    ticks_procesados = 0
    ultimo_reporte = time.time()
    ultima_recarga_config = time.time()
    ultimo_cierre = 0
    estado_hibernacion = False
    
    # --- SENTIMENT VARIABLES ---
    macro_sentiment_score = 0.0
    last_sentiment_read = 0
    SENTIMENT_FILE_PATH = os.path.join(SCRIPT_DIR, "..", "macro_sentiment.json")

    def read_sentiment():
        try:
            if os.path.exists(SENTIMENT_FILE_PATH):
                with open(SENTIMENT_FILE_PATH, 'r') as f:
                    data = json.load(f)
                    # Expirar sentimiento si es muy viejo (> 5 mins)
                    if time.time() - data.get('timestamp', 0) < 300:
                        return data.get('global_sentiment_score', 0.0)
        except: pass
        return 0.0

    # --- PRE-CALENTAMIENTO DEL DETECTOR DE REGIMEN ---
    print(f"{Fore.YELLOW}[SISTEMA] Pre-calentando detector de regimen de mercado...")
    historico = db.obtener_datos_entrenamiento(3000)
    if not historico.empty:
        for precio in historico['mid_price']:
            regimen_actual = regime_detector.update(precio)
        print(f"{Fore.CYAN}[SISTEMA] Detector listo ({len(historico)} ticks). Regimen actual: {regimen_actual}")
    # -------------------------------------------------
    
    # Conexion Multiplexada (Depth + AggTrade + Liquidations para ETH y BookTicker para BTC)
    url = f"wss://fstream.binance.com/stream?streams={SYMBOL_WSS}@depth10@100ms/{SYMBOL_WSS}@aggTrade/{SYMBOL_WSS}@forceOrder/btcusdt@bookTicker"

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        print(f"{Fore.GREEN}[ONLINE] Simulador ETH Activo. Desconectado de tu dinero real.")
        
        # Variables BTC
        btc_prev_mid = None
        btc_trend = 0.0

        while True:
            try:
                # Leer sentimiento cada 5 segundos
                if time.time() - last_sentiment_read > 5:
                    loop = asyncio.get_event_loop()
                    macro_sentiment_score = await loop.run_in_executor(None, read_sentiment)
                    last_sentiment_read = time.time()

                # Actualizar MTF Context cada 60s
                if time.time() - last_mtf_update > 60:
                    loop = asyncio.get_event_loop()
                    mtf_res = await loop.run_in_executor(None, fetch_mtf_data, SYMBOL_WSS.upper())
                    if mtf_res[0] is not None:
                        ema_15m, rsi_5m, atr_5m = mtf_res
                    else:
                        atr_5m = 0.0
                    last_mtf_update = time.time()

                if time.time() - ultima_recarga_config > 30:
                    cargar_configuracion()
                    ultima_recarga_config = time.time()

                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"\n{Fore.RED}[ALERTA ZOMBIE] Reiniciando conexion simulada...")
                    raise asyncio.TimeoutError("Zombie Timeout")

                json_data = json.loads(msg)
                
                if 'stream' not in json_data:
                    continue
                stream_name = json_data['stream']
                data = json_data['data']
                
                if stream_name == 'btcusdt@bookTicker':
                    btc_mid = (float(data['b']) + float(data['a'])) / 2
                    if btc_prev_mid is not None:
                        # Calculamos el trend como la variacion en ticks suavizada
                        btc_delta = (btc_mid - btc_prev_mid) / btc_prev_mid
                        btc_trend = 0.1 * btc_delta + 0.9 * btc_trend # EMA muy rapida del trend de BTC
                    btc_prev_mid = btc_mid
                    continue
                
                if '@aggTrade' in stream_name:
                    qty = float(data['q'])
                    if data['m']: cvd_vol -= qty  # Venta agresiva
                    else: cvd_vol += qty          # Compra agresiva
                    continue
                    
                elif '@forceOrder' in stream_name:
                    o = data['o']
                    qty = float(o['q'])
                    if o['S'] == 'BUY': liq_shorts += qty   # Liquidaron a un oso
                    else: liq_longs += qty                  # Liquidaron a un toro
                    continue
                
                elif '@depth10' in stream_name:
                    # --- DECAIMIENTO Y EMA DE DATOS AGRESIVOS ---
                    liq_longs *= 0.999
                    liq_shorts *= 0.999
                    
                    alpha_cvd = 2 / (10 + 1)
                    cvd_ema = alpha_cvd * cvd_vol + (1 - alpha_cvd) * cvd_ema
                    cvd_vol = 0.0 # Reset tras el tick

                if time.time() - ultimo_reporte > 300:
                    print(f"\n[ESTADO] PnL Neto Simulado: ${pnl_acumulado:.2f} | Trades: {trades_totales}")
                    ultimo_reporte = time.time()
                
                bids = np.array(data['b'], dtype=float)
                asks = np.array(data['a'], dtype=float)
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                current_vol_bid = np.sum(bids[:,1])
                current_vol_ask = np.sum(asks[:,1])
                imbalance = (current_vol_bid - current_vol_ask) / (current_vol_bid + current_vol_ask)

                # --- CALCULO DE OFI (Order Flow Imbalance) ---
                e_b_t = 0.0
                if prev_best_bid is not None:
                    if best_bid > prev_best_bid:
                        e_b_t = current_vol_bid
                    elif best_bid == prev_best_bid:
                        e_b_t = current_vol_bid - prev_vol_bid
                    elif best_bid < prev_best_bid:
                        e_b_t = -prev_vol_bid
                else:
                    e_b_t = current_vol_bid

                e_a_t = 0.0
                if prev_best_ask is not None:
                    if best_ask < prev_best_ask:
                        e_a_t = current_vol_ask
                    elif best_ask == prev_best_ask:
                        e_a_t = current_vol_ask - prev_vol_ask
                    elif best_ask > prev_best_ask:
                        e_a_t = -prev_vol_ask
                else:
                    e_a_t = current_vol_ask

                ofi_t = e_b_t - e_a_t

                # --- CALCULO DE EMA ITERATIVA ---
                ofi_ema_5_t = alpha_5 * ofi_t + (1 - alpha_5) * prev_ofi_ema_5
                ofi_ema_15_t = alpha_15 * ofi_t + (1 - alpha_15) * prev_ofi_ema_15
                
                ema_15m_dist = (mid_price - ema_15m) / ema_15m if ema_15m else 0.0
                if pd.isna(rsi_5m): rsi_5m = 50.0

                # Fallbacks if local vars are not initialized
                _atr = atr_5m if 'atr_5m' in locals() else 0.0

                features = {
                    'mid_price': mid_price, 'imbalance': imbalance, 
                    'spread': best_ask-best_bid, 'wall_gap': asks[9][0]-bids[9][0],
                    'vol_total': current_vol_bid + current_vol_ask,
                    'best_bid': best_bid, 'best_ask': best_ask,
                    'ofi': ofi_t,
                    'ofi_ema_5': ofi_ema_5_t,
                    'ofi_ema_15': ofi_ema_15_t,
                    'cvd': cvd_ema,
                    'liq_longs': liq_longs,
                    'liq_shorts': liq_shorts,
                    'ema_15m_dist': ema_15m_dist,
                    'rsi_5m': rsi_5m,
                    'btc_trend': btc_trend,
                    'atr_5m': _atr,
                    'macro_sentiment': macro_sentiment_score
                }

                # --- ACTUALIZAR ESTADO PREVIO ---
                prev_best_bid = best_bid
                prev_vol_bid = current_vol_bid
                prev_best_ask = best_ask
                prev_vol_ask = current_vol_ask
                prev_ofi_ema_5 = ofi_ema_5_t
                prev_ofi_ema_15 = ofi_ema_15_t
                
                ticks_procesados += 1
                
                spread_pct = (best_ask - best_bid) / mid_price
                regimen_actual = regime_detector.update(mid_price)
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, db.guardar_tick, features) 
                await loop.run_in_executor(None, db.purgar_datos_viejos)
                precision = await loop.run_in_executor(None, ia.entrenar, db)
                
                prob_up, prob_down = ia.predecir(features)
                
                if ticks_procesados % 10 == 0:
                    col = Fore.GREEN if imbalance > 0 else Fore.RED
                    regime_str = f"[{regimen_actual}] " if regimen_actual != "NORMAL" else ""
                    
                    hora_actual_utc_print = datetime.utcnow().hour
                    en_horario = HORA_INICIO_OP <= hora_actual_utc_print < HORA_FIN_OP
                    
                    if not en_horario and posicion is None:
                        if not estado_hibernacion:
                            print(f"\n{Fore.YELLOW}[SISTEMA] Mercado en bajo volumen. Hibernando hasta las {HORA_INICIO_OP}:00 UTC...{Style.RESET_ALL}")
                            estado_hibernacion = True
                    else:
                        if estado_hibernacion:
                            print(f"\n{Fore.GREEN}[SISTEMA] Despertando de hibernacion. Retomando operativa...{Style.RESET_ALL}")
                            estado_hibernacion = False
                            
                        if not ia.trained:
                            estado_str = f"[CALIBRANDO IA] Recolectando datos..."
                        elif posicion:
                            pnl_actual_pct = (best_bid - precio_entrada) / precio_entrada if posicion == 'LONG' else (precio_entrada - best_ask) / precio_entrada
                            pnl_actual_usd = (MONTO_USDT * LEVERAGE) * pnl_actual_pct
                            color_pnl = Fore.GREEN if pnl_actual_usd > 0 else Fore.RED
                            estado_str = f"{posicion} (In: {precio_entrada:.2f}) | PnL: {color_pnl}{pnl_actual_pct*100:.2f}% (${pnl_actual_usd:.3f}){Style.RESET_ALL}"
                        elif time.time() - ultimo_cierre < cooldown_actual:
                            seg_restantes = int(cooldown_actual - (time.time() - ultimo_cierre))
                            estado_str = f"[COOLING] {seg_restantes}s"
                        else:
                            estado_str = "ESPERANDO"
                        
                        hora_actual = datetime.now().strftime("%H:%M:%S")
                        print(f"\r\033[2K{hora_actual} | ETH {mid_price:.2f} | IMB: {col}{imbalance:.2f}{Style.RESET_ALL} | UP:{prob_up:.2f} DN:{prob_down:.2f} | {regime_str}{estado_str}", end='', flush=True)

                # --- LOGICA DE TRADING (SIMULADA) ---
                if posicion is None and ia.trained:
                    if time.time() - ultimo_cierre < cooldown_actual:
                        continue 

                    if regimen_actual in ["SHOCK", "RANGO", "CALIBRANDO"]:
                        confirmaciones_long = 0
                        confirmaciones_short = 0
                        continue

                    hora_actual_utc = datetime.utcnow().hour
                    if hora_actual_utc < HORA_INICIO_OP or hora_actual_utc >= HORA_FIN_OP:
                        confirmaciones_long = 0
                        confirmaciones_short = 0
                        continue

                    spread_aceptable = spread_pct <= SPREAD_MAXIMO_PCT

                    if prob_up > UMBRAL_CONFIANZA_IA and imbalance > UMBRAL_IMBALANCE and ofi_t > OFI_THRESHOLD and ofi_ema_5_t > OFI_EMA_5_THRESHOLD and spread_aceptable:
                        confirmaciones_long += 1
                        confirmaciones_short = 0
                    elif prob_down > UMBRAL_CONFIANZA_IA and imbalance < -UMBRAL_IMBALANCE and ofi_t < -OFI_THRESHOLD and ofi_ema_5_t < -OFI_EMA_5_THRESHOLD and spread_aceptable:
                        confirmaciones_short += 1
                        confirmaciones_long = 0
                    else:
                        confirmaciones_long = 0
                        confirmaciones_short = 0

                    if confirmaciones_long >= TICKS_CONFIRMACION:
                        print(f"\n{Fore.GREEN}[SIM] SENAL LONG CONFIRMADA ({TICKS_CONFIRMACION} ticks continuos)")
                        posicion = 'LONG'
                        precio_entrada = best_ask # Simula orden TAKER cruzando el spread
                        confirmaciones_long = 0
                        max_pnl_pct = 0.0
                        
                        # --- POSITION SIZING DINAMICO ---
                        monto_invertido = MONTO_USDT
                        if prob_up > 0.95: monto_invertido = MONTO_USDT * 1.5   # Aumento agresivo
                        elif prob_up > 0.90: monto_invertido = MONTO_USDT * 1.2 # Aumento leve
                        if monto_invertido > MONTO_USDT: print(f"{Fore.CYAN}⚖️ Apalancando ${monto_invertido:.2f} USDT por Confianza IA alta.")
                        
                        timestamp_entrada = time.time()
                        guardar_estado_simulacion(posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, prob_up)
                    elif confirmaciones_short >= TICKS_CONFIRMACION:
                        print(f"\n{Fore.RED}[SIM] SENAL SHORT CONFIRMADA ({TICKS_CONFIRMACION} ticks continuos)")
                        posicion = 'SHORT'
                        precio_entrada = best_bid # Simula orden TAKER cruzando el spread
                        confirmaciones_short = 0
                        max_pnl_pct = 0.0
                        
                        # --- POSITION SIZING DINAMICO ---
                        monto_invertido = MONTO_USDT
                        if prob_down > 0.95: monto_invertido = MONTO_USDT * 1.5   # Aumento agresivo
                        elif prob_down > 0.90: monto_invertido = MONTO_USDT * 1.2 # Aumento leve
                        if monto_invertido > MONTO_USDT: print(f"{Fore.CYAN}⚖️ Apalancando ${monto_invertido:.2f} USDT por Confianza IA alta.")
                        
                        timestamp_entrada = time.time()
                        guardar_estado_simulacion(posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, prob_down)

                # B) SALIDAS SIMULADAS
                elif posicion in ['LONG', 'SHORT']:
                    pnl_pct = (best_bid - precio_entrada) / precio_entrada if posicion == 'LONG' else (precio_entrada - best_ask) / precio_entrada
                    precio_salida = best_bid if posicion == 'LONG' else best_ask
                    
                    if pnl_pct > max_pnl_pct:
                        max_pnl_pct = pnl_pct
                        
                    should_close = False
                    motivo = ""
                    
                    # 1. SALIDA DE EMERGENCIA (AI REVERSAL)
                    if ia.trained and not should_close:
                        # IA Modo Panico: Cortar rapido si estamos en perdidas y la senal en contra es fuerte
                        if pnl_pct < 0:
                            if posicion == 'LONG' and prob_down > 0.85: confirmaciones_reversal += 1
                            elif posicion == 'SHORT' and prob_up > 0.85: confirmaciones_reversal += 1
                            else: confirmaciones_reversal = 0
                        else:
                            # Modo Normal: Exigir volumen para cerrar ganancia prematuramente
                            if posicion == 'LONG' and prob_down > 0.80 and imbalance < -0.30: confirmaciones_reversal += 1
                            elif posicion == 'SHORT' and prob_up > 0.80 and imbalance > 0.30: confirmaciones_reversal += 1
                            else: confirmaciones_reversal = 0
                            
                        # Solo cerramos si la reversion es muy sostenida (aprox 2 segundos) para evitar cierres falsos
                        TICKS_REVERSAL = 20
                        if confirmaciones_reversal >= TICKS_REVERSAL:
                            should_close = True
                            motivo = "AI_REVERSAL"
                            confirmaciones_reversal = 0

                    # 2. SALIDA INTELIGENTE Y ANTI-AVARICIA (TRAILING STOP)
                    if not should_close:
                        if pnl_pct >= TAKE_PROFIT_PCT * 1.25: 
                            should_close = True
                            motivo = "HARD_TP"
                        elif pnl_pct >= TAKE_PROFIT_PCT:
                            if posicion == 'LONG' and prob_up > 0.70 and imbalance > 0.10: pass 
                            elif posicion == 'SHORT' and prob_down > 0.70 and imbalance < -0.10: pass 
                            else:
                                should_close = True
                                motivo = "SMART_TP"
                        else:
                            sl_dinamico = -STOP_LOSS_PCT
                            
                            # --- LOGICA DE TRAILING STOP ---
                            ACTIVACION_TS_PCT = TAKE_PROFIT_PCT * 0.85 # Exigir 85% de ganancia antes de asegurar
                            DISTANCIA_TS_PCT = TAKE_PROFIT_PCT * 0.2 # Persecución más ajustada para no ahogar el R:R
                            
                            if max_pnl_pct >= ACTIVACION_TS_PCT:
                                # Garantizar cubrir comisiones + spread como minimo de ganancia
                                piso_ganancia = ROUND_TRIP_FEE + SPREAD_MAXIMO_PCT
                                sl_dinamico = max(piso_ganancia, max_pnl_pct - DISTANCIA_TS_PCT)
                                
                            if pnl_pct <= sl_dinamico:
                                should_close = True
                                motivo = "TRAILING_STOP" if max_pnl_pct >= ACTIVACION_TS_PCT else "SL"
                        
                    if should_close:
                        color = Fore.GREEN if motivo == "HARD_TP" else (Fore.YELLOW if motivo == "SMART_TP" else (Fore.LIGHTRED_EX if motivo == "AI_REVERSAL" else (Fore.CYAN if motivo == "TRAILING_STOP" else Fore.MAGENTA)))
                        pnl_bruto_usd = (monto_invertido * LEVERAGE) * pnl_pct
                        costo_fees_usd = (monto_invertido * LEVERAGE) * ROUND_TRIP_FEE
                        pnl_neto_usd = pnl_bruto_usd - costo_fees_usd
                        
                        pnl_acumulado += pnl_neto_usd
                        trades_totales += 1
                        
                        # --- CORTACIRCUITOS DINÁMICO ---
                        if pnl_neto_usd > 0:
                            rachas_perdidas = 0
                            cooldown_actual = 60
                        else:
                            rachas_perdidas += 1
                            if rachas_perdidas >= 4:
                                cooldown_actual = 3600 # Reducido a 1 hora
                                print(f"\n{Fore.RED}🛑 [HIBERNACION] 4 perdidas consecutivas. Mercado TOXICO. Bot apagado por 1 hora.")
                            else:
                                cooldown_actual = 60 * rachas_perdidas # Cooldown mucho más corto (1 min, 2 min...)
                                print(f"\n{Fore.RED}⚠️ [CORTACIRCUITOS] Racha perdedora: {rachas_perdidas}. Bot pausado por {cooldown_actual/60:.0f} minutos.")
                        # --------------------------------
                        
                        print(f"\n{color}[CERRADO] {motivo} ETH | PnL Bruto: {pnl_pct*100:.2f}% | NETO: ${pnl_neto_usd:.3f}")
                        registrar_trade_log(posicion, precio_entrada, precio_salida, pnl_pct, pnl_neto_usd)
                        
                        posicion = None
                        max_pnl_pct = 0.0 
                        ultimo_cierre = time.time()
                        confirmaciones_reversal = 0
                        timestamp_entrada = 0.0
                    guardar_estado_simulacion(posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                print(f"\n{Fore.RED}[DESCONEXION] Conexion perdida. Reconectando en 2s...")
                await asyncio.sleep(2)
                break
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Loop Simulacion ETH: {repr(e)}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

if __name__ == "__main__":
    db = GestorDB(DB_NAME)
    ia = CerebroIA()
    try:
        while True:
            try:
                asyncio.run(main_loop(db, ia))
            except Exception as e:
                print(f"[ERROR] Reiniciando loop principal por: {e}")
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n[SISTEMA] Simulador ETH Detenido. Guardando datos finales...")
        db.close()