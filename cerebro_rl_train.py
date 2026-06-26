import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import pandas as pd
import numpy as np
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Por favor instala las dependencias de RL: pip install stable-baselines3[extra] gymnasium")
    sys.exit(1)
from colorama import Fore, Style, init

from cerebro_rl_env import TradingEnv
from bot_core import GestorDB

init(autoreset=True)

DB_NAME = "cerebro_eth.db"
MODEL_PATH = "modelo_rl_eth" # stable-baselines3 agrega el .zip automáticamente
TERMINAL_LOG_PATH = "log_terminal_data.json"

def recalcular_features(df):
    """Reconstruye las features de OFI y EMAs en caso de que falten en la BD antigua"""
    for col in ['cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'macro_sentiment']:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0)

    if 'ofi' not in df.columns or 'ofi_ema_5' not in df.columns:
        df['vol_bid'] = df['vol_total'] * (1 + df['imbalance']) / 2
        df['vol_ask'] = df['vol_total'] * (1 - df['imbalance']) / 2
        df['e_b'] = 0.0
        df['e_a'] = 0.0
        
        df.loc[df['best_bid'] > df['best_bid'].shift(1), 'e_b'] = df['vol_bid']
        df.loc[df['best_bid'] == df['best_bid'].shift(1), 'e_b'] = df['vol_bid'] - df['vol_bid'].shift(1)
        df.loc[df['best_bid'] < df['best_bid'].shift(1), 'e_b'] = -df['vol_bid'].shift(1)
        
        df.loc[df['best_ask'] < df['best_ask'].shift(1), 'e_a'] = df['vol_ask']
        df.loc[df['best_ask'] == df['best_ask'].shift(1), 'e_a'] = df['vol_ask'] - df['vol_ask'].shift(1)
        df.loc[df['best_ask'] > df['best_ask'].shift(1), 'e_a'] = -df['vol_ask'].shift(1)
        
        df['ofi'] = df['e_b'] - df['e_a']
        
        span_5, span_15 = 5, 15
        alpha_5, alpha_15 = 2 / (span_5 + 1), 2 / (span_15 + 1)
        df['ofi_ema_5'] = 0.0
        df['ofi_ema_15'] = 0.0
        
        if len(df) > 0:
            df.loc[0, 'ofi_ema_5'] = df.loc[0, 'ofi']
            df.loc[0, 'ofi_ema_15'] = df.loc[0, 'ofi']
            for i in range(1, len(df)):
                df.loc[i, 'ofi_ema_5'] = alpha_5 * df.loc[i, 'ofi'] + (1 - alpha_5) * df.loc[i-1, 'ofi_ema_5']
                df.loc[i, 'ofi_ema_15'] = alpha_15 * df.loc[i, 'ofi'] + (1 - alpha_15) * df.loc[i-1, 'ofi_ema_15']

    return df.dropna().reset_index(drop=True)

def main():
    print(f"{Fore.MAGENTA}=====================================================")
    print(f"{Fore.MAGENTA}  MOTOR RL INICIADO (STABLE-BASELINES3 + PYTORCH)    ")
    print(f"{Fore.MAGENTA}=====================================================")
    
    print(f"{Fore.CYAN}[INFO] Cargando datos históricos de {DB_NAME}...")
    db = GestorDB(DB_NAME, TERMINAL_LOG_PATH)
    # Cargar suficientes ticks para el entrenamiento
    df = db.obtener_datos_entrenamiento(100000)
    db.close()
    
    if df.empty or len(df) < 1000:
        print(f"{Fore.RED}[ERROR] No hay suficientes datos en la BD para entrenar.")
        sys.exit(1)
        
    # Limpiar posibles NaNs de las transformaciones y recalcular OFI si falta
    df = recalcular_features(df)
    
    print(f"{Fore.GREEN}[SUCCESS] Datos procesados: {len(df)} ticks.")
    
    # Crear el entorno
    env = TradingEnv(df)
    
    # Stable-Baselines recomienda vectorizar el entorno
    vec_env = DummyVecEnv([lambda: env])
    
    # Configurar el Agente PPO
    print(f"\n{Fore.YELLOW}[TRAIN] Construyendo modelo PPO...")
    # Usamos MlpPolicy (Red Neuronal Perceptrón Multicapa)
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99, # Factor de descuento
        device="cuda", # Forzamos a usar la GPU
        tensorboard_log="./tensorboard_rl_logs/"
    )
    
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"{Fore.BLUE}[INFO] Cargando modelo previo para continuar entrenamiento...")
        model.set_parameters(MODEL_PATH)
        
    print(f"\n{Fore.GREEN}🚀 Iniciando entrenamiento (100,000 timesteps)...")
    try:
        model.learn(total_timesteps=100000, progress_bar=True)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[WARN] Entrenamiento interrumpido por el usuario.")
        
    print(f"\n{Fore.GREEN}✅ Entrenamiento finalizado. Guardando modelo en {MODEL_PATH}")
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
