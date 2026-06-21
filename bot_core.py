import json
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import sqlite3
from colorama import Fore, Style, init
from datetime import datetime
import warnings
from sklearn.utils.class_weight import compute_sample_weight

init(autoreset=True)
warnings.filterwarnings('ignore')

# --- CONFIGURACION GENERAL (SHARED) ---
# These will be loaded from config_params.json or set as defaults
UMBRAL_CONFIANZA_IA = 0.85
TAKE_PROFIT_PCT = 0.005
STOP_LOSS_PCT = 0.01
UMBRAL_IMBALANCE = 0.25
OFI_THRESHOLD = 0.01
OFI_EMA_5_THRESHOLD = 0.01

# PARAMETROS FIJOS (SHARED)
VENTANA_APRENDIZAJE = 50000
MONTO_USDT = 8.0
LEVERAGE = 20
MAX_PERDIDA_DIARIA = -2.5
COOLDOWN_SEGUNDOS = 300
HORA_INICIO_OP = 0   # 00:00 UTC
HORA_FIN_OP = 24     # 24:00 UTC (Todo el dia)
ROUND_TRIP_FEE = 0.0007
TICKS_CONFIRMACION = 1
SPREAD_MAXIMO_PCT = 0.0003
ACTIVACION_BE_PCT = 0.0030
ACTIVACION_TS_PCT = TAKE_PROFIT_PCT * 0.80
DISTANCIA_TS_PCT = TAKE_PROFIT_PCT * 0.25

# --- FUNCIONES DE LOG Y CONFIG ---
def cargar_configuracion(config_file_path):
    global TAKE_PROFIT_PCT, STOP_LOSS_PCT, UMBRAL_IMBALANCE, UMBRAL_CONFIANZA_IA, OFI_THRESHOLD, OFI_EMA_5_THRESHOLD
    try:
        if not os.path.exists(config_file_path):
            print(f"{Fore.YELLOW}[WARNING] No se encontro {config_file_path}. Creando configuracion por defecto...")
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
            with open(config_file_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return True

        with open(config_file_path, 'r') as f:
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
        log_terminal_event("ERROR", "CONFIG", f"Error cargando configuracion: {str(e)}", config_file_path)
        return False

def log_terminal_event(level, event_type, message, terminal_log_file_path, metadata=None):
    """Guarda eventos estructurados en JSON Lines para depuracion y analisis IA."""
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "type": event_type,
            "msg": message,
            "meta": metadata or {}
        }
        with open(terminal_log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] No se pudo escribir en el log terminal: {e}")

def registrar_trade_log(tipo, entry_price, exit_price, pnl_pct, pnl_usd, log_file_path, terminal_log_file_path):
    """Registra un resumen del trade en el log de trades y en el log terminal JSON."""
    try:
        hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resultado = "WIN" if pnl_usd > 0 else "LOSS"
        linea = f"[{hora}] {resultado} | {tipo} | IN: {entry_price:.2f} | OUT: {exit_price:.2f} | PNL%: {pnl_pct*100:.3f}% | NETO: ${pnl_usd:.4f}\n"
        with open(log_file_path, "a") as f:
            f.write(linea)

        # Also log to the structured terminal log
        log_terminal_event("INFO", "TRADE_SUMMARY", f"Trade {resultado}: {tipo} from {entry_price:.2f} to {exit_price:.2f}", terminal_log_file_path, {
            "type": tipo,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "result": resultado
        })
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Error escribiendo log de trade: {e}")
        log_terminal_event("ERROR", "TRADE_LOG_ERROR", f"Error escribiendo log de trade: {str(e)}", terminal_log_file_path)

def guardar_estado_simulacion(state_file_path, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido=6.0, rachas_perdidas=0, timestamp_entrada=0.0, confianza_ia=0.0):
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
        temp_file = state_file_path + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(state, f)

        for _ in range(10):
            try:
                os.replace(temp_file, state_file_path)
                break
            except PermissionError:
                time.sleep(0.01)
    except Exception as e:
        print(f"{Fore.RED}[ERROR] No se pudo guardar estado simulado: {e}")
        log_terminal_event("ERROR", "STATE_SAVE_ERROR", f"No se pudo guardar estado simulado: {str(e)}", state_file_path)

def cargar_estado_simulacion(state_file_path):
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                st = json.load(f)
                return st.get("posicion"), st.get("precio_entrada", 0.0), st.get("max_pnl_pct", 0.0), st.get("pnl_acumulado", 0.0), st.get("trades_totales", 0), st.get("monto_invertido", 6.0), st.get("rachas_perdidas", 0), st.get("timestamp_entrada", 0.0), st.get("confianza_ia", 0.0)
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Error cargando estado simulado: {e}. Iniciando con estado por defecto.")
            log_terminal_event("WARNING", "STATE_LOAD_ERROR", f"Error cargando estado simulado: {str(e)}", state_file_path)
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
    def __init__(self, db_name, terminal_log_file_path, buffer_size=500):
        self.db_name = db_name
        self.terminal_log_file_path = terminal_log_file_path
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
        # Add columns if they don't exist
        columns_to_add = {
            "cvd": "REAL DEFAULT 0.0",
            "liq_longs": "REAL DEFAULT 0.0",
            "liq_shorts": "REAL DEFAULT 0.0",
            "ema_15m_dist": "REAL DEFAULT 0.0",
            "rsi_5m": "REAL DEFAULT 50.0",
            "btc_trend": "REAL DEFAULT 0.0",
            "atr_5m": "REAL DEFAULT 0.0",
            "macro_sentiment": "REAL DEFAULT 0.0"
        }
        for col, col_type in columns_to_add.items():
            try:
                self.cursor.execute(f"ALTER TABLE mercado_ticks ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    log_terminal_event("ERROR", "DB_ALTER_TABLE", f"Error al añadir columna {col}: {e}", self.terminal_log_file_path)
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

        try:
            self.cursor.executemany('''
                INSERT INTO mercado_ticks (timestamp, mid_price, imbalance, spread, wall_gap, vol_total, best_bid, best_ask, ofi, ofi_ema_5, ofi_ema_15, cvd, liq_longs, liq_shorts, ema_15m_dist, rsi_5m, btc_trend, atr_5m, macro_sentiment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self.buffer)
            self.conn.commit()
            self.buffer.clear()
        except Exception as e:
            log_terminal_event("ERROR", "DB_FLUSH_ERROR", f"Error al vaciar buffer de DB: {str(e)}", self.terminal_log_file_path)


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
            log_terminal_event("WARNING", "DB_PURGE", f"Error purgando DB: {str(e)}", self.terminal_log_file_path)

    def close(self):
        self.flush()
        self.conn.close()

# --- DATOS MACRO (MULTI-TIMEFRAME) ---
def fetch_mtf_data(symbol, terminal_log_file_path):
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
        log_terminal_event("ERROR", "MTF_FETCH_ERROR", f"Error fetching MTF data: {str(e)}", terminal_log_file_path)
        return None, None, None

# --- CEREBRO IA ---
class CerebroIA:
    """
    Motor de Inferencia y Entrenamiento XGBoost con aceleracion hardware (CUDA).
    Implementa Zero-Pandas inference y Time-Based Windowing.
    """
    def __init__(self, model_path, terminal_log_file_path):
        self.model_path = model_path
        self.terminal_log_file_path = terminal_log_file_path
        self.model = xgb.XGBClassifier(
            n_estimators=80, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
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
                    # This is a common issue with older XGBoost models or specific save formats
                    # We assume classes are 0, 1, 2 for multi:softprob with num_class=3
                    self.model.classes_ = np.array([0, 1, 2])
                self.trained = True
                print(f"{Fore.GREEN}[IA] Modelo ETH cargado desde disco. Listo para operar.")
            except Exception as e:
                print(f"{Fore.YELLOW}[IA WARNING] Error cargando modelo previo: {e}")
                log_terminal_event("WARNING", "IA_LOAD_ERROR", f"Error cargando modelo previo: {str(e)}", self.terminal_log_file_path)


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

        MIN_PROFIT_PCT = 0.0025 # Sincronizado con el bot real (0.25%)
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
            log_terminal_event("WARNING", "IA_TRAINING_SKIPPED", "No hay suficientes clases unicas para entrenar la IA.", self.terminal_log_file_path)
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
        # --- PENALIZACIÓN DE RUIDO ---
        pesos[y_fit.values == 0] *= 0.5
        if missing_classes:
            pesos[-len(missing_classes):] = 0.0

        try:
            self.model.fit(X_fit, y_fit, sample_weight=pesos)
            self.trained = True
            self.last_train_time = time.time()
            self.model.save_model(self.model_path)
            return self.model.score(X_fit, y_fit)
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Falla en entrenamiento: {e}")
            log_terminal_event("ERROR", "IA_TRAINING", f"Fallo al entrenar modelo: {str(e)}", self.terminal_log_file_path)
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
        # Ensure that the classes are correctly mapped
        # XGBoost's predict for multi:softprob returns probabilities in order of class labels
        # If classes_ is not set, it defaults to 0, 1, 2.
        # If the model was trained with different class order, this needs to be handled.
        # Assuming 0=neutral, 1=up, 2=down based on target creation in entrenar
        if hasattr(self.model, 'classes_'):
            for i, clase in enumerate(self.model.classes_):
                if clase == 1: prob_up = probs[i]
                elif clase == 2: prob_down = probs[i]
        else: # Fallback if classes_ attribute is missing (e.g., older model versions)
            # Assuming default order: probs[0] for class 0, probs[1] for class 1, probs[2] for class 2
            prob_up = probs[1] if len(probs) > 1 else 0.0
            prob_down = probs[2] if len(probs) > 2 else 0.0

        return prob_up, prob_down

def read_sentiment(sentiment_file_path, terminal_log_file_path):
    try:
        if os.path.exists(sentiment_file_path):
            with open(sentiment_file_path, 'r') as f:
                data = json.load(f)
                # Expirar sentimiento si es muy viejo (> 5 mins)
                if time.time() - data.get('timestamp', 0) < 300:
                    return data.get('global_sentiment_score', 0.0)
    except Exception as e:
        log_terminal_event("ERROR", "SENTIMENT_READ_ERROR", f"Error leyendo archivo de sentimiento: {str(e)}", terminal_log_file_path)
    return 0.0