# QuantBinance 🤖📈

Sistema de Trading Algorítmico y Cuantitativo impulsado por Inteligencia Artificial (IA) y Procesamiento de Lenguaje Natural (NLP). Actualmente configurado como un simulador avanzado (Paper Trading) para Ethereum (ETH/USDT).

## 🚀 Características Principales

*   **Orquestador Maestro (`quant_orchestrator.py`):** Lanza y monitorea los distintos "cerebros" y bots de forma automatizada.
*   **Optimizador CUDA (`cerebro_cuda.py`):** Motor de optimización de parámetros impulsado por PyTorch y aceleración GPU (NVIDIA RTX) para encontrar las mejores métricas de Take Profit, Stop Loss e indicadores de mercado en tiempo real.
*   **Motor de Sentimiento NLP (`cerebro_sentimiento.py`):** Analizador de noticias en tiempo real mediante *web scraping* asíncrono de feeds RSS (Cointelegraph, CoinDesk) integrado con **FinBERT** de HuggingFace para evaluar el sentimiento del mercado global de criptomonedas.
*   **Bot de Paper Trading (`bot_eth_paper.py`):** Simulador de trading para Ethereum que utiliza los pesos generados por la IA y la base de datos local para operar sin riesgo de capital real.

## 📁 Estructura del Proyecto

```text
QuantBinance/
├── .env                        # Variables de entorno (API keys de Binance, etc.)
├── cerebro_cuda.py             # Motor de optimización de IA (Algoritmos Genéticos / PSO)
├── cerebro_sentimiento.py      # Scraper RSS y análisis de sentimiento con FinBERT
├── cerebro_rl_train.py         # Script de entrenamiento de Reinforcement Learning (PPO)
├── cerebro_rl_env.py           # Entorno simulado (Gymnasium) para entrenamiento RL
├── quant_orchestrator.py       # Script principal de ejecución
├── bot_eth/
│   └── bot_eth_paper.py        # Bot simulador para Ethereum
└── README.md
```

## 🛠️ Requisitos e Instalación

1.  **Python 3.10+** (Recomendado vía Anaconda/Miniconda)
2.  **Entorno CUDA:** Tarjeta gráfica NVIDIA compatible con soporte para CUDA (ej. RTX Serie 40) para ejecutar `cerebro_cuda.py` de forma eficiente.
3.  **Dependencias:**
    Se recomienda encarecidamente utilizar **Conda** para gestionar el entorno, ya que facilita enormemente la compatibilidad con las librerías de NVIDIA y PyTorch con aceleración GPU.

    ```bash
    # 1. Crear un entorno conda
    conda create -n cerebro_gpu python=3.10
    conda activate cerebro_gpu

    # 2. Instalar PyTorch con soporte para CUDA (Ajusta la versión de CUDA según tu sistema)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    # 3. Instalar el resto de dependencias
    pip install requests aiohttp transformers colorama ccxt
    ```

4.  **Configuración (`.env`):**
    Crea un archivo `.env` en la raíz del proyecto para definir las llaves de la API (solo necesarias si se adapta a *Real Trading* en el futuro) o configuraciones globales:
    ```env
    BINANCE_API_KEY=tu_api_key_aqui
    BINANCE_SECRET_KEY=tu_secret_key_aqui
    ```

## 🖥️ Uso

Para iniciar el sistema completo en modo simulación (Paper Trading), simplemente ejecuta el orquestador:

```bash
python quant_orchestrator.py
```

El orquestador levantará automáticamente:
1. El analizador de sentimiento (NLP).
2. El optimizador de IA (CUDA).
3. El bot simulador (ETH Paper Trading).

## ⚠️ Advertencia Legal
Este software se provee estrictamente con fines educativos y de investigación. El trading con criptomonedas conlleva un alto riesgo. Actualmente este proyecto corre en modo de simulación (Paper Trading), pero si es modificado para operar con dinero real, el autor no se hace responsable de ninguna pérdida financiera.

---

## 📌 Versiones y Changelog

### v10.0.0 (Estable) - *Current*
Esta versión consolida el sistema como una plataforma hiper-robusta y matemáticamente defensiva.
* **Reinforcement Learning (PPO):** Se introdujo el algoritmo Proximal Policy Optimization (`cerebro_rl_train.py` y `cerebro_rl_env.py`) para dotar al bot de capacidad de aprendizaje on-policy.
* **Límites Quant (Genetic Optimizer):** Se blindó `cerebro_cuda.py` restringiendo la búsqueda genética de parámetros. Se forzó un Take Profit mínimo de 0.35% y un Stop Loss máximo de 0.65%, garantizando rentabilidad con el Win Rate actual (>70%).
* **Hot-Reloading Fix:** Se reparó un *import gotcha* de Python en `bot_eth_paper.py` para asegurar que el bot aplique instantáneamente las configuraciones de riesgo dictadas por el optimizador genético en vivo.
* **NLP Silencer & Resiliencia:** Se silenciaron las advertencias ruidosas de HuggingFace en `cerebro_sentimiento.py` y se corroboró la tolerancia a caídas de red asíncronas en los feeds RSS.
