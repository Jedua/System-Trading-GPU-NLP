import subprocess
import time
import sys
import os
import site
from colorama import Fore, Style, init

init(autoreset=True)

def run_process(script_path, name):
    """Inicia un script Python en un proceso separado y devuelve el objeto del proceso."""
    print(f"{Fore.CYAN}[ORQUESTADOR] Iniciando {name} ({script_path})...")
    try:
        # Usamos sys.executable para asegurarnos de que se use el mismo intérprete Python
        process = subprocess.Popen([sys.executable, script_path])
        print(f"{Fore.GREEN}[ORQUESTADOR] {name} iniciado con PID: {process.pid}")
        return process
    except Exception as e:
        print(f"{Fore.RED}[ORQUESTADOR ERROR] No se pudo iniciar {name}: {e}")
        return None

def _encontrar_dlls_nvidia():
    """
    Busca las librerias CUDA instaladas por pip/conda y las anade al PATH.
    Esto es un workaround comun en Windows para problemas de visibilidad de Numba CUDA.
    """
    rutas_posibles = site.getsitepackages()
    rutas_posibles.append(site.getusersitepackages())
    
    ruta_encontrada = None
    for ruta in rutas_posibles:
        candidata = os.path.join(ruta, "nvidia", "cuda_runtime", "bin")
        if os.path.exists(candidata):
            ruta_encontrada = candidata
            break
            
    if ruta_encontrada:
        os.environ['PATH'] = ruta_encontrada + os.pathsep + os.environ['PATH']
        print(f"{Fore.BLUE}[ORQUESTADOR] Parche de PATH aplicado: {ruta_encontrada}")
        return True
    return False

def main():
    print(f"{Fore.MAGENTA}=============================================")
    print(f"{Fore.MAGENTA}      MASTER QUANT ORCHESTRATOR INICIADO     ")
    print(f"{Fore.MAGENTA}=============================================")

    # --- CONFIGURACION DE ENTORNO ---
    MODO_PRODUCCION = False # Cambiar a True para usar dinero real (bot_eth_live.py)
    
    if MODO_PRODUCCION:
        print(f"{Fore.RED}{Style.BRIGHT}!!! ADVERTENCIA: MODO PRODUCCION ACTIVADO. SE USARA DINERO REAL !!!{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}MODO PAPER TRADING (Simulacion Segura).{Style.RESET_ALL}")

    # --- VERIFICACION DE ENTORNO CUDA ---
    try:
        # Intentamos aplicar el parche de PATH antes de la verificacion de Numba
        _encontrar_dlls_nvidia() 
        from numba import cuda
        if not cuda.is_available():
            print(f"{Fore.RED}[ORQUESTADOR ERROR] GPU no detectada por Numba en este entorno.")
            print(f"{Fore.YELLOW}[SOLUCION] Ejecute este script desde un terminal con el entorno Conda correcto activado (ej: 'conda activate cerebro_gpu').")
            sys.exit(1)
        gpu = cuda.get_current_device()
        print(f"{Fore.GREEN}[ORQUESTADOR] Entorno CUDA verificado. GPU disponible: {gpu.name.decode('utf-8')}")
    except ImportError:
        print(f"{Fore.RED}[ORQUESTADOR ERROR] Numba no esta instalado en este entorno.")
        print(f"{Fore.YELLOW}[SOLUCION] Ejecute este script desde un terminal con el entorno Conda correcto activado.")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}[ORQUESTADOR ERROR] Falla en verificacion CUDA: {e}")
        sys.exit(1)

    # Rutas absolutas a tus scripts
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CEREBRO_CUDA_PATH = os.path.join(BASE_DIR, "cerebro_cuda.py")
    
    if MODO_PRODUCCION:
        BOT_ETH_PATH = os.path.join(BASE_DIR, "bot_eth", "bot_eth_live.py")
        bot_name_label = "Bot ETH (LIVE/REAL)"
    else:
        BOT_ETH_PATH = os.path.join(BASE_DIR, "bot_eth", "bot_eth_paper.py")
        bot_name_label = "Bot ETH (Simulador)"
        
    SENTIMENT_PATH = os.path.join(BASE_DIR, "cerebro_sentimiento.py")

    # Iniciar el motor de sentimiento NLP
    sentiment_process = run_process(SENTIMENT_PATH, "Cerebro Sentimiento (NLP)")

    # Iniciar el optimizador PSO en un proceso
    cerebro_process = run_process(CEREBRO_CUDA_PATH, "Cerebro CUDA (Optimizador)")

    # Esperar un poco para que el cerebro inicialice y posiblemente genere la config inicial
    print(f"{Fore.BLUE}[ORQUESTADOR] Esperando 10 segundos para que los motores inicialicen...")
    time.sleep(10)

    # Iniciar el bot en otro proceso
    bot_process = run_process(BOT_ETH_PATH, bot_name_label)

    if not cerebro_process or not bot_process or not sentiment_process:
        print(f"{Fore.RED}[ORQUESTADOR] Uno o más procesos fallaron al iniciar. Deteniendo...")
        # Intentar terminar cualquier proceso que haya iniciado
        if cerebro_process and cerebro_process.poll() is None: cerebro_process.terminate()
        if bot_process and bot_process.poll() is None: bot_process.terminate()
        if sentiment_process and sentiment_process.poll() is None: sentiment_process.terminate()
        sys.exit(1)

    try:
        while True:
            # Monitorear si los procesos siguen vivos
            if sentiment_process.poll() is not None:
                print(f"{Fore.RED}[ORQUESTADOR] Cerebro Sentimiento ha terminado. Reiniciando...")
                sentiment_process = run_process(SENTIMENT_PATH, "Cerebro Sentimiento (NLP)")
            if cerebro_process.poll() is not None:
                print(f"{Fore.RED}[ORQUESTADOR] Cerebro CUDA ha terminado. Reiniciando...")
                cerebro_process = run_process(CEREBRO_CUDA_PATH, "Cerebro CUDA (Optimizador)")
            if bot_process.poll() is not None:
                print(f"{Fore.RED}[ORQUESTADOR] Bot ETH ha terminado. Reiniciando...")
                bot_process = run_process(BOT_ETH_PATH, bot_name_label)
            time.sleep(5) # Chequear cada 5 segundos
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[ORQUESTADOR] Apagando procesos controlados...")
        if sentiment_process.poll() is None: sentiment_process.terminate()
        if cerebro_process.poll() is None: cerebro_process.terminate()
        if bot_process.poll() is None: bot_process.terminate()
        print(f"{Fore.GREEN}[ORQUESTADOR] Todos los procesos terminados.")

if __name__ == "__main__":
    main()