import os
import sys
import pandas as pd
from colorama import init, Fore, Style

init(autoreset=True)

def analizar_logs(rutas_logs):
    trades = []
    
    for ruta in rutas_logs:
        if not os.path.exists(ruta):
            continue
            
        # Determinar la moneda por el nombre del archivo
        moneda = "ETH"
        if "ada" in ruta.lower(): moneda = "ADA"
        elif "sol" in ruta.lower(): moneda = "SOL"
        
        with open(ruta, 'r') as f:
            lineas = f.readlines()
            for linea in lineas:
                linea = linea.strip()
                if not linea or "---" in linea or not linea.startswith("["):
                    continue
                    
                try:
                    partes = linea.split(" | ")
                    tiempo_y_resultado = partes[0].split("] ")
                    timestamp = tiempo_y_resultado[0].replace("[", "")
                    resultado = tiempo_y_resultado[1]
                    tipo = partes[1]
                    
                    # Extraer PNL% para fallback
                    pnl_str = [p for p in partes if "PNL%:" in p][0].replace("PNL%: ", "").replace("%", "")
                    pnl_pct = float(pnl_str) / 100.0
                    
                    pnl_neto = 0.0
                    if "NETO: $" in linea:
                        neto_str = [p for p in partes if "NETO: $" in p][0].replace("NETO: $", "")
                        pnl_neto = float(neto_str)
                        resultado = "WIN" if pnl_neto > 0 else "LOSS"
                    else:
                        # Reconstruccion para logs antiguos de ADA (sin NETO)
                        inversion_aprox = 8.0 * 20.0
                        gross = inversion_aprox * pnl_pct
                        fees = inversion_aprox * 0.0010 
                        pnl_neto = gross - fees
                        resultado = "WIN" if pnl_neto > 0 else "LOSS"

                    trades.append({
                        "timestamp": timestamp,
                        "moneda": moneda,
                        "resultado": resultado,
                        "tipo": tipo,
                        "pnl_neto": pnl_neto
                    })
                except Exception as e:
                    pass 

    if not trades:
        print(f"{Fore.YELLOW}[INFO] No hay trades validos en los logs proporcionados.")
        return

    df = pd.DataFrame(trades)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # --- LISTADO VISUAL ---
    print(f"\n{Fore.CYAN}==================================================")
    print(f"{Fore.CYAN}              HISTORIAL DE TRADES                 ")
    print(f"{Fore.CYAN}==================================================")
    for _, row in df.iterrows():
        color_moneda = Fore.MAGENTA if row['moneda'] == 'ETH' else Fore.BLUE
        color_res = Fore.GREEN if row['pnl_neto'] > 0 else Fore.RED
        print(f"[{row['timestamp']}] {color_moneda}[{row['moneda']}]{Style.RESET_ALL} {row['tipo']:<5} -> {color_res}{row['resultado']:<4} | NETO: ${row['pnl_neto']:.4f}{Style.RESET_ALL}")
    
    # Calculos Estadisticos
    total_trades = len(df)
    wins = len(df[df['pnl_neto'] > 0])
    losses = len(df[df['pnl_neto'] <= 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    
    gross_profit = df[df['pnl_neto'] > 0]['pnl_neto'].sum()
    gross_loss = abs(df[df['pnl_neto'] < 0]['pnl_neto'].sum())
    
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    net_profit = df['pnl_neto'].sum()
    
    # Calculo de Max Drawdown (Caida maxima desde el pico)
    df['balance_acumulado'] = df['pnl_neto'].cumsum()
    df['pico_maximo'] = df['balance_acumulado'].cummax()
    df['drawdown'] = df['balance_acumulado'] - df['pico_maximo']
    max_drawdown = df['drawdown'].min()

    # Desglose por moneda
    pnl_eth = df[df['moneda'] == 'ETH']['pnl_neto'].sum()
    pnl_ada = df[df['moneda'] == 'ADA']['pnl_neto'].sum()

    # Reporte en Consola
    print(f"\n{Fore.CYAN}==================================================")
    print(f"{Fore.CYAN}          REPORTE DE RENDIMIENTO COMBINADO        ")
    print(f"{Fore.CYAN}==================================================")
    print(f"Total Operaciones  : {total_trades}")
    print(f"Ganadoras (WINS)   : {Fore.GREEN}{wins}{Style.RESET_ALL}")
    print(f"Perdedoras (LOSS)  : {Fore.RED}{losses}{Style.RESET_ALL}")
    print(f"Win Rate Global    : {Fore.YELLOW}{win_rate:.2f}%{Style.RESET_ALL}")
    print(f"--------------------------------------------------")
    print(f"PnL ETH            : {Fore.GREEN if pnl_eth > 0 else Fore.RED}${pnl_eth:.4f}{Style.RESET_ALL}")
    print(f"PnL ADA            : {Fore.GREEN if pnl_ada > 0 else Fore.RED}${pnl_ada:.4f}{Style.RESET_ALL}")
    print(f"--------------------------------------------------")
    print(f"Ganancia Bruta     : {Fore.GREEN}${gross_profit:.4f}{Style.RESET_ALL}")
    print(f"Perdida Bruta      : {Fore.RED}-${gross_loss:.4f}{Style.RESET_ALL}")
    
    pf_color = Fore.GREEN if profit_factor >= 1.5 else (Fore.YELLOW if profit_factor >= 1.0 else Fore.RED)
    print(f"Profit Factor      : {pf_color}{profit_factor:.2f}{Style.RESET_ALL}")
    
    net_color = Fore.GREEN if net_profit > 0 else Fore.RED
    print(f"PnL Neto Total     : {net_color}${net_profit:.4f}{Style.RESET_ALL}")
    
    print(f"Maximum Drawdown   : {Fore.RED}${max_drawdown:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Archivos a leer: logs reales
    logs_a_leer = [
        os.path.join(script_dir, "bot_eth", "trading_log.txt"),
        os.path.join(script_dir, "bot_ada", "real_trading_log_ada.txt")
    ]
    
    analizar_logs(logs_a_leer)