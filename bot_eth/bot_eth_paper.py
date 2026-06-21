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

from bot_core import cargar_configuracion, log_terminal_event, registrar_trade_log, guardar_estado_simulacion, cargar_estado_simulacion, MarketRegime, GestorDB, fetch_mtf_data, CerebroIA, read_sentiment, UMBRAL_CONFIANZA_IA, TAKE_PROFIT_PCT, STOP_LOSS_PCT, UMBRAL_IMBALANCE, OFI_THRESHOLD, OFI_EMA_5_THRESHOLD, VENTANA_APRENDIZAJE, MONTO_USDT, LEVERAGE, MAX_PERDIDA_DIARIA, COOLDOWN_SEGUNDOS, HORA_INICIO_OP, HORA_FIN_OP, ROUND_TRIP_FEE, TICKS_CONFIRMACION, SPREAD_MAXIMO_PCT, ACTIVACION_BE_PCT, ACTIVACION_TS_PCT, DISTANCIA_TS_PCT

# --- CONFIGURACION ---
SYMBOL_WSS = 'ethusdt'  
DB_NAME = os.path.join(SCRIPT_DIR, "cerebro_eth.db")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "..", "config_params.json")
PAPER_LOG_FILE = os.path.join(SCRIPT_DIR, "paper_trading_log.txt")
PAPER_STATE_FILE = os.path.join(SCRIPT_DIR, "sim_state_eth.json")
PAPER_MODEL_FILE = os.path.join(SCRIPT_DIR, "modelo_ia_eth.json")
PAPER_TERMINAL_LOG_FILE = os.path.join(SCRIPT_DIR, "log_terminal_data.json")

init(autoreset=True)
warnings.filterwarnings('ignore')

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

    cargar_configuracion(CONFIG_FILE)
    print(f"[CONFIG] IA {UMBRAL_CONFIANZA_IA} | IMB {UMBRAL_IMBALANCE} | TP {TAKE_PROFIT_PCT*100:.2f}% | SL {STOP_LOSS_PCT*100:.2f}% | OFI {OFI_THRESHOLD} | EMA5 {OFI_EMA_5_THRESHOLD}")
    
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
                precision = await loop.run_in_executor(None, ia.entrenar, db) # From bot_core

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
                if posicion is None and ia.trained and not estado_hibernacion:
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

                    # --- FILTROS ESTRICTOS DE SEGURIDAD (HARD FILTERS) ---
                    filtro_sentiment_long = macro_sentiment_score > -0.20 # From bot_core
                    filtro_sentiment_short = macro_sentiment_score < 0.20
                    
                    filtro_tendencia_long = ema_15m_dist >= -0.001 and rsi_5m < 70.0
                    filtro_tendencia_short = ema_15m_dist <= 0.001 and rsi_5m > 30.0

                    if prob_up > UMBRAL_CONFIANZA_IA and imbalance > UMBRAL_IMBALANCE and ofi_t > OFI_THRESHOLD and ofi_ema_5_t > OFI_EMA_5_THRESHOLD and spread_aceptable and filtro_sentiment_long and filtro_tendencia_long:
                        confirmaciones_long += 1
                        confirmaciones_short = 0
                    elif prob_down > UMBRAL_CONFIANZA_IA and imbalance < -UMBRAL_IMBALANCE and ofi_t < -OFI_THRESHOLD and ofi_ema_5_t < -OFI_EMA_5_THRESHOLD and spread_aceptable and filtro_sentiment_short and filtro_tendencia_short:
                        confirmaciones_short += 1
                        confirmaciones_long = 0
                    else:
                        confirmaciones_long = 0
                        confirmaciones_short = 0

                    if confirmaciones_long >= TICKS_CONFIRMACION:
                        print(f"\n{Fore.GREEN}[SIM] SENAL LONG CONFIRMADA ({TICKS_CONFIRMACION} ticks continuos)")
                        posicion = 'LONG'
                        precio_entrada = best_ask # Simula orden MARKET TAKER (paga spread)
                        confirmaciones_long = 0
                        max_pnl_pct = 0.0
                        
                        # --- POSITION SIZING DINAMICO ---
                        monto_invertido = MONTO_USDT
                        if prob_up > 0.95: monto_invertido = MONTO_USDT * 1.5   # Aumento agresivo
                        elif prob_up > 0.90: monto_invertido = MONTO_USDT * 1.2 # Aumento leve
                        if monto_invertido > MONTO_USDT: print(f"{Fore.CYAN}⚖️ Apalancando ${monto_invertido:.2f} USDT por Confianza IA alta.")
                        
                        timestamp_entrada = time.time()
                        
                        # Guardar radiografia de la entrada
                        log_terminal_event("INFO", "TRADE_ENTRY", f"LONG a {precio_entrada}", PAPER_TERMINAL_LOG_FILE, {
                            "prob_up": round(prob_up, 4), "prob_down": round(prob_down, 4),
                            "imbalance": round(imbalance, 4), "ofi": round(ofi_t, 2), 
                            "cvd_ema": round(cvd_ema, 2), "rsi": round(rsi_5m, 2),
                            "spread": spread_pct, "monto": monto_invertido,
                            "regimen": regimen_actual, "sentiment": macro_sentiment_score
                        })
                        
                        guardar_estado_simulacion(PAPER_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, prob_up)
                    elif confirmaciones_short >= TICKS_CONFIRMACION:
                        print(f"\n{Fore.RED}[SIM] SENAL SHORT CONFIRMADA ({TICKS_CONFIRMACION} ticks continuos)")
                        posicion = 'SHORT'
                        precio_entrada = best_bid # Simula orden MARKET TAKER (paga spread)
                        confirmaciones_short = 0
                        max_pnl_pct = 0.0
                        
                        # --- POSITION SIZING DINAMICO ---
                        monto_invertido = MONTO_USDT
                        if prob_down > 0.95: monto_invertido = MONTO_USDT * 1.5   # Aumento agresivo
                        elif prob_down > 0.90: monto_invertido = MONTO_USDT * 1.2 # Aumento leve
                        if monto_invertido > MONTO_USDT: print(f"{Fore.CYAN}⚖️ Apalancando ${monto_invertido:.2f} USDT por Confianza IA alta.")
                        
                        timestamp_entrada = time.time()
                        
                        # Guardar radiografia de la entrada
                        log_terminal_event("INFO", "TRADE_ENTRY", f"SHORT a {precio_entrada}", PAPER_TERMINAL_LOG_FILE, {
                            "prob_up": round(prob_up, 4), "prob_down": round(prob_down, 4),
                            "imbalance": round(imbalance, 4), "ofi": round(ofi_t, 2), 
                            "cvd_ema": round(cvd_ema, 2), "rsi": round(rsi_5m, 2),
                            "spread": spread_pct, "monto": monto_invertido,
                            "regimen": regimen_actual, "sentiment": macro_sentiment_score
                        })
                        
                        guardar_estado_simulacion(PAPER_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, prob_down)

                # B) SALIDAS SIMULADAS
                elif posicion in ['LONG', 'SHORT']:
                    # Simular Market Taker Exit (Cruzar el spread)
                    # En producción real, los SL, Trailing y Pánico son a mercado.
                    # LONG sale vendiendo al Bid. SHORT sale comprando al Ask.
                    precio_salida = best_bid if posicion == 'LONG' else best_ask
                    pnl_pct = (precio_salida - precio_entrada) / precio_entrada if posicion == 'LONG' else (precio_entrada - precio_salida) / precio_entrada
                    
                    if pnl_pct > max_pnl_pct:
                        max_pnl_pct = pnl_pct
                        
                    should_close = False
                    motivo = ""
                    
                    duracion_trade = time.time() - timestamp_entrada if timestamp_entrada > 0 else 0
                    
                    # 0. TIME STOP: Evitar desangrarse lentamente (Max 1 hora)
                    if duracion_trade > 900 and not should_close: # Reduced from 3600s (1 hour) to 900s (15 minutes)
                        if pnl_pct < (TAKE_PROFIT_PCT * 0.5): # Si no estamos claramente ganando, abortar
                            should_close = True
                            motivo = "TIME_STOP"
                    
                    # 1. SALIDA DE EMERGENCIA (AI PÁNICO)
                    if ia.trained and not should_close:
                        # Filtro de paciencia: No entrar en pánico en los primeros 10 segundos
                        if duracion_trade > 10:
                            # IA Modo Panico: Cortar rapido si estamos en perdidas y la senal en contra es fuerte
                            if pnl_pct < 0: # From bot_core
                                if posicion == 'LONG' and prob_down > 0.85: confirmaciones_reversal += 1
                                elif posicion == 'SHORT' and prob_up > 0.85: confirmaciones_reversal += 1
                                else: confirmaciones_reversal = 0
                                ticks_necesarios = 40 # Cortar perdidas tras ~4s de confirmacion constante
                            else:
                                # Modo Paciente: Exigir mucha fuerza para cerrar una ganancia prematuramente # From bot_core
                                if posicion == 'LONG' and prob_down > 0.85 and imbalance < -0.40: confirmaciones_reversal += 1
                                elif posicion == 'SHORT' and prob_up > 0.85 and imbalance > 0.40: confirmaciones_reversal += 1
                                else: confirmaciones_reversal = 0
                                
                                ticks_necesarios = 100 # Ser paciente con las ganancias (~10s)
                                
                            if confirmaciones_reversal >= ticks_necesarios:
                                should_close = True
                                motivo = "AI_REVERSAL"
                                confirmaciones_reversal = 0
                        else:
                            confirmaciones_reversal = 0

                    # 2. SALIDA INTELIGENTE Y ANTI-AVARICIA (TRAILING STOP Y BREAK-EVEN)
                    if not should_close:
                        if pnl_pct >= TAKE_PROFIT_PCT * 1.25: # From bot_core
                            should_close = True
                            motivo = "HARD_TP"
                        elif pnl_pct >= TAKE_PROFIT_PCT: # From bot_core
                            if posicion == 'LONG' and prob_up > 0.70 and imbalance > 0.10: pass 
                            elif posicion == 'SHORT' and prob_down > 0.70 and imbalance < -0.10: pass 
                            else:
                                should_close = True
                                motivo = "SMART_TP"
                        else:
                            sl_dinamico = -STOP_LOSS_PCT
                            
                            # --- 2.1 BREAK-EVEN AUTOMATICO ---
                            # Si ganamos más de un 0.35%, protegemos la operacion cobrando comisiones minimo
                            ACTIVACION_BE_PCT = 0.0035 
                            if max_pnl_pct >= ACTIVACION_BE_PCT:
                                sl_dinamico = ROUND_TRIP_FEE * 1.5 # Colchon extra para garantizar 0 perdidas reales
                            
                            # --- 2.2 TRAILING STOP ---
                            ACTIVACION_TS_PCT = TAKE_PROFIT_PCT * 0.80 # Exigir 80% de ganancia antes de perseguir
                            DISTANCIA_TS_PCT = TAKE_PROFIT_PCT * 0.25 # Distancia para no ahogar el trade
                            
                            if max_pnl_pct >= ACTIVACION_TS_PCT:
                                piso_ganancia = max(ROUND_TRIP_FEE * 1.5, sl_dinamico)
                                sl_dinamico = max(piso_ganancia, max_pnl_pct - DISTANCIA_TS_PCT)
                                
                            if pnl_pct <= sl_dinamico:
                                should_close = True
                                if max_pnl_pct >= ACTIVACION_TS_PCT: motivo = "TRAILING_STOP"
                                elif max_pnl_pct >= ACTIVACION_BE_PCT: motivo = "BREAK_EVEN"
                                else: motivo = "SL"
                        
                    if should_close:
                        color = Fore.GREEN if motivo == "HARD_TP" else (Fore.YELLOW if motivo == "SMART_TP" else (Fore.LIGHTRED_EX if motivo in ["AI_REVERSAL", "TIME_STOP"] else (Fore.CYAN if motivo in ["TRAILING_STOP", "BREAK_EVEN"] else Fore.MAGENTA)))
                        pnl_bruto_usd = (monto_invertido * LEVERAGE) * pnl_pct
                        costo_fees_usd = (monto_invertido * LEVERAGE) * ROUND_TRIP_FEE
                        pnl_neto_usd = pnl_bruto_usd - costo_fees_usd
                        
                        pnl_acumulado += pnl_neto_usd
                        trades_totales += 1
                        
                        # --- CORTACIRCUITOS DINÁMICO ---
                        if pnl_neto_usd > 0:
                            rachas_perdidas = 0
                            cooldown_actual = 10 # Reducido a 10s para no perder ráfagas de la tendencia
                        else:
                            rachas_perdidas += 1
                            if rachas_perdidas >= 4:
                                cooldown_actual = 3600 # Reducido a 1 hora
                                print(f"\n{Fore.RED}🛑 [HIBERNACION] 4 perdidas consecutivas. Mercado TOXICO. Bot apagado por 1 hora.")
                                log_terminal_event("WARNING", "RISK_CORTACIRCUITOS", "Mercado toxico, apagando por 1h", PAPER_TERMINAL_LOG_FILE)
                            else:
                                cooldown_actual = 60 * rachas_perdidas # Cooldown mucho más corto (1 min, 2 min...)
                                print(f"\n{Fore.RED}⚠️ [CORTACIRCUITOS] Racha perdedora: {rachas_perdidas}. Bot pausado por {cooldown_actual/60:.0f} minutos.")
                                log_terminal_event("WARNING", "RISK_PAUSA", f"Pausa tras perdida. Cooldown: {cooldown_actual}s", PAPER_TERMINAL_LOG_FILE)
                        # --------------------------------
                        
                        print(f"\n{color}[CERRADO] {motivo} ETH | PnL Bruto: {pnl_pct*100:.2f}% | NETO: ${pnl_neto_usd:.3f}")
                        
                        log_terminal_event("INFO", "TRADE_EXIT", f"Cierre por {motivo}", PAPER_TERMINAL_LOG_FILE, {
                            "exit_price": precio_salida, "pnl_pct": round(pnl_pct, 5), 
                            "pnl_usd": round(pnl_neto_usd, 4), "max_pnl": round(max_pnl_pct, 5),
                            "duracion_seg": round(time.time() - timestamp_entrada, 1)
                        })
                        
                        registrar_trade_log(posicion, precio_entrada, precio_salida, pnl_pct, pnl_neto_usd, PAPER_LOG_FILE, PAPER_TERMINAL_LOG_FILE)
                        
                        posicion = None
                        max_pnl_pct = 0.0 
                        ultimo_cierre = time.time()
                        confirmaciones_reversal = 0
                        timestamp_entrada = 0.0
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
    ia = CerebroIA(PAPER_MODEL_FILE, PAPER_TERMINAL_LOG_FILE)
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