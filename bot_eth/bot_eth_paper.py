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

# Add parent directory to sys.path to allow absolute imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from bot_core import cargar_configuracion, log_terminal_event, registrar_trade_log, guardar_estado_simulacion, cargar_estado_simulacion, MarketRegime, GestorDB, fetch_mtf_data, CerebroRL, read_sentiment, GestorExperiencia, RiskManager, VENTANA_APRENDIZAJE, MONTO_USDT, LEVERAGE, MAX_PERDIDA_DIARIA, COOLDOWN_SEGUNDOS, HORA_INICIO_OP, HORA_FIN_OP, ROUND_TRIP_FEE, TICKS_CONFIRMACION, SPREAD_MAXIMO_PCT, ACTIVACION_BE_PCT, ACTIVACION_TS_PCT
import bot_core

# --- CONFIGURACION ---
SYMBOL_WSS = 'ethusdt'  
DB_NAME = os.path.join(PARENT_DIR, "cerebro_eth.db")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "..", "config_params.json")
PAPER_LOG_FILE = os.path.join(PARENT_DIR, "paper_trading_log.txt")
PAPER_STATE_FILE = os.path.join(PARENT_DIR, "sim_state_eth.json")
PAPER_MODEL_FILE = os.path.join(SCRIPT_DIR, "..", "modelo_rl_eth.zip")
PAPER_TERMINAL_LOG_FILE = os.path.join(PARENT_DIR, "log_terminal_data.json")

init(autoreset=True)
warnings.filterwarnings('ignore')

# --- MOTOR PRINCIPAL ---
async def main_loop(db, ia, exp, risk_manager):
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

    cargar_configuracion(CONFIG_FILE)
    print(f"[CONFIG] IA {bot_core.UMBRAL_CONFIANZA_IA} | IMB {bot_core.UMBRAL_IMBALANCE} | TP {bot_core.TAKE_PROFIT_PCT*100:.2f}% | SL {bot_core.STOP_LOSS_PCT*100:.2f}% | OFI {bot_core.OFI_THRESHOLD} | EMA5 {bot_core.OFI_EMA_5_THRESHOLD}")
    
    if not os.path.exists(PAPER_LOG_FILE):
        with open(PAPER_LOG_FILE, "w") as f:
            f.write(f"--- INICIO DE PAPER TRADING LOG ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
    log_terminal_event("INFO", "SYSTEM_START", "Iniciando simulacion ETH.", PAPER_TERMINAL_LOG_FILE)
    print(f"[LOGS] Guardando en: {PAPER_LOG_FILE}")

    # --- VARIABLES DE ESTADO PARA OFI Y EMA ---
    prev_best_bid = None
    prev_vol_bid = None
    prev_best_ask = None
    prev_vol_ask = None
    prev_ofi_ema_5 = 0.0 # Initial value for EMA calculation
    prev_ofi_ema_15 = 0.0 # Initial value for EMA calculation

    # EMA spans
    span_5 = 5
    alpha_5 = 2 / (span_5 + 1)
    span_15 = 15
    alpha_15 = 2 / (span_15 + 1)
    
    regime_detector = MarketRegime() # From bot_core
    posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, confianza_ia = cargar_estado_simulacion(PAPER_STATE_FILE) # From bot_core
    cooldown_actual = rachas_perdidas * COOLDOWN_SEGUNDOS if rachas_perdidas > 0 else 0
    
    confirmaciones_long = 0
    confirmaciones_short = 0
    TICKS_CONFIRMACION = 1 # Reducido para igualar la agresividad del backtest en CUDA
    confirmaciones_reversal = 0
    SPREAD_MAXIMO_PCT = 0.0003 # 3 BPS maximo para evitar deslizamiento fuerte
    
    ticks_procesados = 0
    ultimo_reporte = time.time()
    ultima_recarga_config = time.time()
    ultimo_cierre = 0
    estado_hibernacion = False
    
    # --- SENTIMENT VARIABLES ---
    macro_sentiment_score = 0.0
    last_sentiment_read = 0
    SENTIMENT_FILE_PATH = os.path.join(SCRIPT_DIR, "..", "macro_sentiment.json")

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
                    macro_sentiment_score = await loop.run_in_executor(None, read_sentiment, SENTIMENT_FILE_PATH, PAPER_TERMINAL_LOG_FILE)
                    last_sentiment_read = time.time()

                # Actualizar MTF Context cada 60s
                if time.time() - last_mtf_update > 60:
                    loop = asyncio.get_event_loop()
                    mtf_res = await loop.run_in_executor(None, fetch_mtf_data, SYMBOL_WSS.upper(), PAPER_TERMINAL_LOG_FILE)
                    if mtf_res[0] is not None: # From bot_core
                        ema_15m, rsi_5m, atr_5m = mtf_res
                    else:
                        atr_5m = 0.0
                    last_mtf_update = time.time()

                if time.time() - ultima_recarga_config > 30:
                    cargar_configuracion(CONFIG_FILE) # From bot_core
                    ultima_recarga_config = time.time()

                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"\n{Fore.RED}[ALERTA ZOMBIE] Reiniciando conexion simulada...")
                    log_terminal_event("ERROR", "NETWORK_ZOMBIE", "Timeout en WS de Binance. Reconectando.", PAPER_TERMINAL_LOG_FILE)
                    raise asyncio.TimeoutError("Zombie Timeout") # Re-raise to break and reconnect

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
                
                loop = asyncio.get_event_loop() # From bot_core
                await loop.run_in_executor(None, db.guardar_tick, features) # From bot_core
                await loop.run_in_executor(None, db.purgar_datos_viejos) # From bot_core
                # --- INFERENCIA RL ---
                pos_int = 1 if posicion == 'LONG' else (-1 if posicion == 'SHORT' else 0)
                current_pnl_pct = 0.0
                if pos_int == 1: current_pnl_pct = (best_bid - precio_entrada) / precio_entrada
                elif pos_int == -1: current_pnl_pct = (precio_entrada - best_ask) / precio_entrada

                action = ia.predecir_accion(features, pos_int, current_pnl_pct)
                
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
                        print(f"\r\033[2K{hora_actual} | ETH {mid_price:.2f} | IMB: {col}{imbalance:.2f}{Style.RESET_ALL} | ACT:{action} | {regime_str}{estado_str}", end='', flush=True)

                # --- LOGICA DE TRADING RL (SIMULADA) ---
                if ia.trained and not estado_hibernacion:
                    
                    if time.time() - ultimo_cierre < cooldown_actual:
                        continue
                        
                    # FUNCION DE CIERRE AUXILIAR
                    def cerrar_posicion(motivo="RL_CLOSE"):
                        nonlocal posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, cooldown_actual, ultimo_cierre, timestamp_entrada
                        
                        precio_salida = best_bid if posicion == 'LONG' else best_ask
                        pnl_pct = (precio_salida - precio_entrada) / precio_entrada if posicion == 'LONG' else (precio_entrada - precio_salida) / precio_entrada
                        
                        color = Fore.GREEN if pnl_pct > ROUND_TRIP_FEE else Fore.RED
                        pnl_bruto_usd = (monto_invertido * LEVERAGE) * pnl_pct
                        costo_fees_usd = (monto_invertido * LEVERAGE) * ROUND_TRIP_FEE
                        pnl_neto_usd = pnl_bruto_usd - costo_fees_usd
                        
                        pnl_acumulado += pnl_neto_usd
                        trades_totales += 1
                        
                        # --- GUARDAR SNAPSHOT DE SALIDA (HER) ---
                        exp.guardar_snapshot("CLOSE_" + posicion, precio_salida, pnl_neto_usd, features)
                        
                        # Cortacircuitos
                        if pnl_neto_usd > 0:
                            rachas_perdidas = 0
                            cooldown_actual = 5
                        else:
                            rachas_perdidas += 1
                            if rachas_perdidas >= 4:
                                cooldown_actual = 3600
                                print(f"\n{Fore.RED}🛑 [HIBERNACION] 4 perdidas consecutivas.")
                            else:
                                cooldown_actual = 60 * rachas_perdidas
                        
                        print(f"\n{color}[CERRADO] {motivo} ETH | PnL Bruto: {pnl_pct*100:.2f}% | NETO: ${pnl_neto_usd:.3f}")
                        registrar_trade_log(posicion, precio_entrada, precio_salida, pnl_pct, pnl_neto_usd, PAPER_LOG_FILE, PAPER_TERMINAL_LOG_FILE)
                        
                        posicion = None
                        max_pnl_pct = 0.0 
                        ultimo_cierre = time.time()
                        timestamp_entrada = 0.0

                    # Procesar Acción RL
                    # 1: Open Long
                    if action == 1:
                        if posicion == 'SHORT': cerrar_posicion("RL_REVERSAL")
                        if posicion is None:
                            print(f"\n{Fore.GREEN}[SIM-RL] SENAL LONG")
                            posicion = 'LONG'
                            precio_entrada = best_ask
                            # --- RISK MANAGER DECIDE TAMANO POSICION ---
                            current_lev = risk_manager.calcular_apalancamiento(100.0 + pnl_acumulado) # Asumiendo $100 capital base
                            monto_invertido = MONTO_USDT * (current_lev / LEVERAGE)
                            timestamp_entrada = time.time()
                            # --- GUARDAR SNAPSHOT DE ENTRADA (HER) ---
                            exp.guardar_snapshot("OPEN_LONG", precio_entrada, 0.0, features)
                            
                    # 2: Open Short
                    elif action == 2:
                        if posicion == 'LONG': cerrar_posicion("RL_REVERSAL")
                        if posicion is None:
                            print(f"\n{Fore.RED}[SIM-RL] SENAL SHORT")
                            posicion = 'SHORT'
                            precio_entrada = best_bid
                            # --- RISK MANAGER DECIDE TAMANO POSICION ---
                            current_lev = risk_manager.calcular_apalancamiento(100.0 + pnl_acumulado)
                            monto_invertido = MONTO_USDT * (current_lev / LEVERAGE)
                            timestamp_entrada = time.time()
                            # --- GUARDAR SNAPSHOT DE ENTRADA (HER) ---
                            exp.guardar_snapshot("OPEN_SHORT", precio_entrada, 0.0, features)
                            
                    # 3: Close Position
                    elif action == 3 and posicion is not None:
                        cerrar_posicion("RL_CLOSE")
                        
                    # Cierres Automáticos (Stop Loss, Take Profit, Trailing Stop)
                    if posicion is not None:
                        if current_pnl_pct > max_pnl_pct:
                            max_pnl_pct = current_pnl_pct

                        # 1. Take Profit / Trailing Stop
                        if max_pnl_pct >= bot_core.TAKE_PROFIT_PCT:
                            # Si el precio retrocede un poco desde el máximo, aseguramos ganancia
                            retroceso = max_pnl_pct - current_pnl_pct
                            if retroceso >= (bot_core.TAKE_PROFIT_PCT * 0.25) or current_pnl_pct >= (bot_core.TAKE_PROFIT_PCT * 1.5):
                                cerrar_posicion("TAKE_PROFIT/TS")
                        
                        # 2. Stop Loss de Emergencia
                        elif current_pnl_pct <= -bot_core.STOP_LOSS_PCT:
                            cerrar_posicion("HARD_SL")
                            
                    guardar_estado_simulacion(PAPER_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                print(f"\n{Fore.RED}[DESCONEXION] Conexion perdida. Reconectando en 2s...")
                log_terminal_event("WARNING", "NETWORK_DISCONNECT", "Conexion WS cerrada por el servidor.", PAPER_TERMINAL_LOG_FILE)
                await asyncio.sleep(2)
                break
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR] Loop Simulacion ETH: {repr(e)}")
                log_terminal_event("ERROR", "SYSTEM_EXCEPTION", repr(e), PAPER_TERMINAL_LOG_FILE)
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

if __name__ == "__main__":
    db = GestorDB(DB_NAME, PAPER_TERMINAL_LOG_FILE)
    ia = CerebroRL(PAPER_MODEL_FILE, PAPER_TERMINAL_LOG_FILE)
    exp = GestorExperiencia(os.path.join(PARENT_DIR, "cerebro_experiencia.db"))
    risk_manager = RiskManager(balance_inicial=100.0, max_leverage=LEVERAGE)
    
    try:
        while True:
            try:
                asyncio.run(main_loop(db, ia, exp, risk_manager))
            except Exception as e:
                print(f"[ERROR] Reiniciando loop principal por: {e}")
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n[SISTEMA] Simulador ETH Detenido. Guardando datos finales...")
        db.close()
        exp.close()