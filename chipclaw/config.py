from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
INDEX = DATA / 'index'
MODEL = os.environ.get('CHIPCLAW_MODEL', 'qwen3:8b')
OLLAMA = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
EMBEDDING = os.environ.get('CHIPCLAW_EMBEDDING', 'intfloat/multilingual-e5-small')
EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L6-v2'
RERANKER_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

SOURCES = {
    'stm32f103': {'title': 'STM32F103x8 / STM32F103xB datasheet', 'url': 'https://www.st.com/resource/en/datasheet/stm32f103cb.pdf'},
    'rp2040': {'title': 'RP2040 datasheet', 'url': 'https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf'},
}

# Isolated leave-document-out experiment. Main corpus/index stays frozen.
HOLDOUT_SOURCES = {
    'tmp117': {'title': 'TMP117 digital temperature sensor datasheet',
               'url': 'https://www.ti.com/lit/ds/symlink/tmp117.pdf'},
}
HOLDOUT_INDEX = DATA / 'holdout_index'

# Larger benchmark uses a separate index; historical indexes remain intact.
EXPANSION_SOURCES = {
    'ina226': {'title': 'INA226 current voltage and power monitor datasheet',
               'url': 'https://www.ti.com/lit/ds/symlink/ina226.pdf'},
    'bme280': {'title': 'BME280 humidity pressure temperature sensor datasheet',
               'url': 'https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf'},
}
DOCUMENT_SOURCES = {**SOURCES, **HOLDOUT_SOURCES, **EXPANSION_SOURCES}
EXPANDED_INDEX = DATA / 'expanded_index'
