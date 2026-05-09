# LSTM Stock Price Predictor

Projeto de Machine Learning ponta a ponta que prevê o **preço de fechamento de ações da bolsa de valores** utilizando uma rede neural **LSTM (Long Short-Term Memory)**, servida via **API RESTful com FastAPI**.

Desenvolvido para o **FIAP CHALLENGE_04**.

---

## Estrutura do Projeto

```
.
├── data_processing.py   # Pipeline de coleta e pré-processamento dos dados
├── model.py             # Arquitetura LSTM, treinamento e avaliação
├── export.py            # Salvamento e carregamento de artefatos (modelo + scaler)
├── app.py               # API RESTful com FastAPI
├── requirements.txt     # Dependências Python com versões fixadas
├── Procfile             # Configuração para deploy no Heroku/Render
├── start.sh             # Script de inicialização para servidores Linux
└── artifacts/           # Criado automaticamente após o treino
    ├── lstm_model.keras         # Modelo LSTM treinado
    ├── scaler.pkl               # MinMaxScaler serializado
    └── training_metadata.json   # Métricas e metadados do treino
```

---

## Visão Geral da Arquitetura

```
BRAPI (brapi.dev) — API financeira brasileira
        │
        ▼
┌─────────────────────┐
│  data_processing.py │  → Download, limpeza, normalização, sequências 3D
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│     model.py        │  → Arquitetura LSTM, treinamento, MAE/RMSE/MAPE
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│     export.py       │  → Salva .keras + .pkl + .json em artifacts/
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      app.py         │  → API FastAPI: POST /predict, GET /health
└─────────────────────┘
```

**Rede LSTM utilizada:**
```
Input(60 timesteps, 1 feature)
  → LSTM(128 unidades, return_sequences=True)
  → Dropout(20%)
  → LSTM(64 unidades)
  → Dropout(20%)
  → Dense(32, ReLU)
  → Dense(1, Linear)  ← preço previsto
```

---

## Pré-requisitos

- **Python 3.10, 3.11 ou 3.12** (recomendado: 3.11)
- `pip` atualizado (`pip install --upgrade pip`)
- Conexão com a internet (para baixar dados via [BRAPI](https://brapi.dev) — API financeira brasileira da B3)

Verifique sua versão:
```bash
python --version
```

---

## Instalação

### 1. Clone ou baixe o projeto

```bash
# Se usar Git:
git clone <url-do-repositorio>
cd Code

# Ou simplesmente navegue até a pasta do projeto:
cd E:\Projetos\FIAP\CHALLENGE_04\Code
```

### 2. Crie e ative um ambiente virtual

É uma boa prática isolar as dependências do projeto para não conflitar com outros projetos Python.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

> Você saberá que o ambiente está ativo quando ver `(.venv)` no início do terminal.

### 3. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Como Trocar o Ativo (Ação)

Abra o arquivo `data_processing.py` e altere as variáveis no início do arquivo:

```python
# Linha ~55 do arquivo data_processing.py

TICKER: str     = "ITUB4"         # ← Troque aqui. SEM sufixo '.SA' (ex: "PETR4", "VALE3")
START_DATE: str = "2018-01-01"    # ← Data de início do histórico
END_DATE: str   = "2024-12-31"    # ← Data de fim do histórico
SEQUENCE_LENGTH: int = 60         # ← Janela temporal (dias usados para prever 1 dia)
BRAPI_TOKEN: str | None = None    # ← Token opcional: https://brapi.dev/dashboard
```

> ⚠️ A BRAPI usa o código da B3 **sem** o sufixo `.SA`. Use `"PETR4"` e não `"PETR4.SA"`.

**Exemplos de tickers válidos (B3):**

| Ativo | Ticker BRAPI |
|---|---|
| Petrobras | `PETR4` |
| Vale | `VALE3` |
| Itaú Unibanco | `ITUB4` |
| Bradesco | `BBDC4` |
| Ambev | `ABEV3` |
| WEG | `WEGE3` |

---

## Treinamento do Modelo

Execute o script de treinamento. Ele realiza **todo o pipeline automaticamente**:
download dos dados → pré-processamento → treinamento → avaliação → salvamento.

```bash
python model.py
```

**O que você verá no terminal:**
```
2026-05-02 14:00:00 [INFO] INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO DE DADOS
2026-05-02 14:00:01 [INFO] Baixando dados do ativo 'ITUB4' de 2018-01-01 até 2024-12-31 via BRAPI...
2026-05-02 14:00:03 [INFO] Download concluído com sucesso na tentativa 1. Shape: (1726, 5)
...
Epoch 1/100 - loss: 0.0043 - val_loss: 0.0051
Epoch 2/100 - loss: 0.0038 - val_loss: 0.0047
...
========================================
MÉTRICAS DE AVALIAÇÃO NO CONJUNTO DE TESTE
  MAE  : 0.8432  (unidades monetárias)
  RMSE : 1.1205  (unidades monetárias)
  MAPE : 2.8741%
========================================
✓ Modelo salvo em: artifacts/lstm_model.keras
✓ Scaler salvo em: artifacts/scaler.pkl
✓ Metadados salvos em: artifacts/training_metadata.json
```

> O treinamento termina automaticamente antes das 100 épocas graças ao **Early Stopping**.

**Artefatos gerados após o treino:**
```
artifacts/
├── lstm_model.keras          # Modelo completo
├── scaler.pkl                # Normalizador de dados
└── training_metadata.json    # Métricas e configurações
```

---

## 🌐 Rodando a API Localmente

Após o treinamento, inicie o servidor:

```bash
python app.py
```

Ou diretamente com uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em:
- **Swagger UI (documentação interativa):** http://localhost:8000/docs
- **ReDoc (documentação alternativa):** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

---

## 🔌 Endpoints da API

### `GET /health` — Verificação de saúde

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "message": "Todos os componentes operacionais."
}
```

---

### `GET /model/info` — Informações do modelo

```bash
curl http://localhost:8000/model/info
```

**Resposta:**
```json
{
  "metrics": {
    "MAE": 0.8432,
    "RMSE": 1.1205,
    "MAPE": 2.8741
  },
  "saved_at": "2026-05-02T14:05:30.123456"
}
```

---

### `POST /predict` — Prever preço de fechamento

Envie uma lista com pelo menos **60 preços históricos** de fechamento.

**Exemplo com `curl`:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [28.50, 28.75, 29.10, 28.90, 29.30, 29.50, 29.20, 28.80,
               29.00, 29.40, 29.60, 29.80, 30.00, 29.70, 29.50, 29.90,
               30.10, 30.30, 30.00, 29.80, 30.20, 30.50, 30.40, 30.60,
               30.80, 31.00, 30.70, 30.50, 30.90, 31.10, 31.30, 31.00,
               30.80, 31.20, 31.50, 31.40, 31.60, 31.80, 32.00, 31.70,
               31.50, 31.90, 32.10, 32.30, 32.00, 31.80, 32.20, 32.50,
               32.40, 32.60, 32.80, 33.00, 32.70, 32.50, 32.90, 33.10,
               33.30, 33.00, 32.80, 33.20],
    "steps": 3
  }'
```

**Resposta:**
```json
{
  "predictions": [33.4521, 33.6789, 33.8012],
  "steps": 3,
  "model_version": "2026-05-02T14:05:30.123456",
  "request_id": "a1b2c3d4"
}
```

**Parâmetros do body:**

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `prices` | `list[float]` | ✅ Sim | Lista de preços de fechamento (mínimo 60) |
| `steps` | `int` | ❌ Não (default: 1) | Quantos dias futuros prever (1 a 30) |

> **Dica:** Use o Swagger UI em http://localhost:8000/docs para testar a API de forma interativa, sem precisar de `curl`.

---

## ☁️ Deploy em Produção (Render / Heroku)

### Opção A — Usar o `Procfile` (Heroku / Render)

1. Faça o commit de todos os arquivos, incluindo a pasta `artifacts/` com o modelo treinado.
2. Conecte o repositório no painel do Heroku ou Render.
3. A plataforma detecta o `Procfile` automaticamente e executa:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2
   ```

### Opção B — Usar o `start.sh` (VPS / servidor Linux)

```bash
# Dê permissão de execução (apenas na primeira vez)
chmod +x start.sh

# Execute
./start.sh
```

O script faz automaticamente:
1. Ativa o ambiente virtual (se existir)
2. Instala dependências via `requirements.txt`
3. Treina o modelo se `artifacts/` estiver vazio
4. Inicia o servidor uvicorn

**Variáveis de ambiente aceitas pelo `start.sh`:**

| Variável | Default | Descrição |
|---|---|---|
| `PORT` | `8000` | Porta do servidor |
| `WORKERS` | `2` | Número de processos uvicorn |

```bash
# Exemplo com variáveis customizadas:
PORT=9000 WORKERS=4 ./start.sh
```

---

## 🔬 Executando Módulos Individualmente

Cada módulo pode ser testado de forma isolada:

```bash
# Testar apenas o pipeline de dados
python data_processing.py

# Treinar o modelo (executa data_processing internamente)
python model.py

# Verificar se os artefatos foram salvos corretamente
python export.py

# Iniciar apenas a API (requer artefatos já existentes)
python app.py
```

---

## 📊 Entendendo as Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---|---|---|
| **MAE** | `mean(|y_real - y_pred|)` | Erro médio em R$/USD. Ex: MAE=0.84 → erra em média R$ 0,84 |
| **RMSE** | `sqrt(mean((y_real - y_pred)²))` | Como MAE, mas penaliza erros grandes. Sempre ≥ MAE |
| **MAPE** | `mean(|erro| / y_real) × 100` | Erro percentual médio. Ex: MAPE=2.87% → erra ~2,87% do preço real |

**Referência de qualidade para MAPE em previsão de ações:**
- < 3% → Excelente
- 3% a 5% → Bom
- 5% a 10% → Aceitável
- \> 10% → Modelo precisa de revisão

---

## 🛠️ Principais Hiperparâmetros para Ajuste

Localizados no início do `model.py`:

```python
LSTM_UNITS_1: int   = 128    # Neurônios na 1ª camada LSTM
LSTM_UNITS_2: int   = 64     # Neurônios na 2ª camada LSTM
DROPOUT_RATE: float = 0.20   # Taxa de regularização (0.0 a 0.5)
LEARNING_RATE: float = 1e-3  # Taxa de aprendizado do otimizador Adam
BATCH_SIZE: int     = 32     # Amostras por atualização de pesos
EPOCHS: int         = 100    # Máximo de épocas (Early Stopping pode parar antes)
PATIENCE: int       = 15     # Épocas sem melhora antes do Early Stopping parar
```

E em `data_processing.py`:

```python
SEQUENCE_LENGTH: int = 60    # Dias de histórico usados para prever 1 dia
TRAIN_SPLIT: float   = 0.80  # 80% treino / 20% teste
```

---

## 🐛 Problemas Comuns e Soluções

**`FileNotFoundError: Modelo não encontrado em 'artifacts/lstm_model.keras'`**
→ Execute o treinamento antes de iniciar a API: `python model.py`

**`ValueError: Nenhum dado retornado para o ticker 'XYZ'`**
→ Verifique o ticker em `data_processing.py`. Use o formato correto (ex: `PETR4.SA` para B3).

**`pip install` falha no TensorFlow no Windows**
→ Certifique-se de usar Python 3.10, 3.11 ou 3.12. O TensorFlow não suporta Python 3.13+.

**API retorna `503 Service Unavailable`**
→ O modelo não foi carregado. Verifique se a pasta `artifacts/` existe e contém os arquivos `.keras` e `.pkl`.

**Treinamento muito lento**
→ Sem GPU, é esperado. Reduza `EPOCHS=30` e `SEQUENCE_LENGTH=30` para testes rápidos.

---

## 📦 Dependências Principais

| Biblioteca | Versão | Uso |
|---|---|---|
| `tensorflow` | 2.17.0 | Rede neural LSTM |
| `fastapi` | 0.115.12 | Framework da API |
| `uvicorn` | 0.34.2 | Servidor ASGI de produção |
| `requests` | 2.32.3 | Chamadas HTTP para a BRAPI (fonte de dados B3) |
| `scikit-learn` | 1.5.2 | Normalização e métricas |
| `pandas` | 2.2.3 | Manipulação de dados |
| `numpy` | 1.26.4 | Operações numéricas |
| `pydantic` | 2.11.4 | Validação dos dados da API |
| `joblib` | 1.4.2 | Serialização do scaler |

---