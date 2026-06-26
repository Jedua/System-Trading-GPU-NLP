import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

class TradingEnv(gym.Env):
    """
    Entorno de Trading personalizado para OpenAI Gymnasium.
    Simula ejecuciones Taker (pagando spread) con comisiones.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=100.0, leverage=20.0, taker_fee=0.0005):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.taker_fee = taker_fee # Comisión por lado (0.05%)
        
        # Acciones:
        # 0: Hold (No hacer nada o mantener posición actual)
        # 1: Open Long (Si ya hay Long, Hold. Si hay Short, cierra Short y abre Long)
        # 2: Open Short (Si ya hay Short, Hold. Si hay Long, cierra Long y abre Short)
        # 3: Close Position (Pasa a Flat)
        self.action_space = spaces.Discrete(4)
        
        # Features del mercado
        self.features_cols = [
            'imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 
            'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 
            'ema_15m_dist', 'rsi_5m', 'macro_sentiment'
        ]
        
        # El observation space incluye features del mercado + estado del agente (posición y PnL)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features_cols) + 2,), dtype=np.float32
        )
        
        # Estado interno
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.trades_count = 0
        self.ticks_in_trade = 0
        
    def _get_obs(self):
        # Tomamos la fila actual
        row = self.df.iloc[self.current_step].copy()
        
        # --- NORMALIZACION PARA ESTABILIZAR RED NEURONAL ---
        # Log scaling para valores absolutos gigantes (volumenes)
        row['vol_total'] = np.log1p(row['vol_total'])
        
        # Reduccion de escala para order flow y deltas
        row['ofi'] = row['ofi'] / 10000.0
        row['ofi_ema_5'] = row['ofi_ema_5'] / 10000.0
        row['ofi_ema_15'] = row['ofi_ema_15'] / 10000.0
        row['cvd'] = row['cvd'] / 10000.0
        row['liq_longs'] = row['liq_longs'] / 10000.0
        row['liq_shorts'] = row['liq_shorts'] / 10000.0
        
        # Centrado estandar [-1, 1]
        row['rsi_5m'] = (row['rsi_5m'] - 50.0) / 50.0
        
        obs = row[self.features_cols].values.astype(np.float32)
        
        # Calculamos PnL flotante (Taker)
        current_pnl_pct = 0.0
        if self.position == 1:
            # Salida Long es vendiendo al Bid
            current_pnl_pct = (row['best_bid'] - self.entry_price) / self.entry_price
        elif self.position == -1:
            # Salida Short es comprando al Ask
            current_pnl_pct = (self.entry_price - row['best_ask']) / self.entry_price
            
        # Agregamos variables de estado del bot
        estado_bot = np.array([self.position, current_pnl_pct], dtype=np.float32)
        return np.concatenate((obs, estado_bot))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trades_count = 0
        self.ticks_in_trade = 0
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        done = False
        row = self.df.iloc[self.current_step]
        
        best_bid = row['best_bid']
        best_ask = row['best_ask']
        
        # Guardar estado previo para la recompensa
        prev_position = self.position

        # --- LÓGICA DE ACCIONES ---
        
        # 0: Hold (Nada cambia)
        if action == 0:
            pass
            
        # 1: Open Long
        elif action == 1:
            if self.position == -1:
                # Cerrar Short existente
                pnl_pct = (self.entry_price - best_ask) / self.entry_price
                net_profit = (pnl_pct * self.leverage) - (self.taker_fee * self.leverage) # Fee solo salida
                reward += net_profit
                self.balance += net_profit * self.balance # Compounding (opcional)
                self.position = 0
                self.ticks_in_trade = 0
                
            if self.position == 0:
                # Abrir Long (Paga el Ask)
                self.position = 1
                self.entry_price = best_ask
                self.trades_count += 1
                reward -= (self.taker_fee * self.leverage) # Fee de entrada inmediato
                
        # 2: Open Short
        elif action == 2:
            if self.position == 1:
                # Cerrar Long existente
                pnl_pct = (best_bid - self.entry_price) / self.entry_price
                net_profit = (pnl_pct * self.leverage) - (self.taker_fee * self.leverage) # Fee solo salida
                reward += net_profit
                self.balance += net_profit * self.balance
                self.position = 0
                self.ticks_in_trade = 0
                
            if self.position == 0:
                # Abrir Short (Paga el Bid)
                self.position = -1
                self.entry_price = best_bid
                self.trades_count += 1
                reward -= (self.taker_fee * self.leverage) # Fee de entrada inmediato
                
        # 3: Close Position
        elif action == 3:
            if self.position == 1:
                pnl_pct = (best_bid - self.entry_price) / self.entry_price
                net_profit = (pnl_pct * self.leverage) - (self.taker_fee * self.leverage) # Fee solo salida
                reward += net_profit
                self.balance += net_profit * self.balance
                self.position = 0
                self.ticks_in_trade = 0
                
            elif self.position == -1:
                pnl_pct = (self.entry_price - best_ask) / self.entry_price
                net_profit = (pnl_pct * self.leverage) - (self.taker_fee * self.leverage) # Fee solo salida
                reward += net_profit
                self.balance += net_profit * self.balance
                self.position = 0
                self.ticks_in_trade = 0

        # --- CASTIGOS Y RECOMPENSAS ADICIONALES ---
        
        # Penalización pequeña por tiempo (Time Decay) para evitar que se quede atascado en un trade eternamente
        if self.position != 0:
            self.ticks_in_trade += 1
            reward -= 0.000005 # Castigo reducido al minimo para evitar el cierre por panico
            
            # Cortacircuitos de seguridad: Stop Loss Forzado del Entorno
            # Si el agente deja que el trade caiga demasiado, lo liquidamos y lo castigamos fuertemente
            current_pnl_pct = 0.0
            if self.position == 1:
                current_pnl_pct = (best_bid - self.entry_price) / self.entry_price
            elif self.position == -1:
                current_pnl_pct = (self.entry_price - best_ask) / self.entry_price
                
            if current_pnl_pct <= -0.015: # -1.5% sin apalancamiento (-30% apalancado)
                reward -= 0.5 # Castigo masivo
                self.position = 0 # Liquidación
                self.ticks_in_trade = 0

        # Castigo por no hacer nada en toda la simulación (para evitar que se rinda y se quede en Hold siempre)
        # Se dará un mini-incentivo por cerrar trades en general si son positivos.

        # --- AVANCE DE TIEMPO ---
        self.current_step += 1
        
        # Comprobar si hemos llegado al final
        if self.current_step >= self.n_steps - 1:
            done = True
            
        # Comprobar bancarrota
        if self.balance <= self.initial_balance * 0.1:
            done = True
            reward -= 1.0 # Penalización por quebrar
            
        obs = self._get_obs()
        info = {
            'balance': self.balance,
            'trades': self.trades_count
        }
        
        # Requerimientos de Gymnasium (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info

    def render(self):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Pos: {self.position} | PnL: {self._get_obs()[-1]*100:.2f}%")
