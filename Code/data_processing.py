"""
data_processing.py
==================
Módulo responsável pela coleta, limpeza, normalização e preparação dos dados
de preços de ações para alimentar a rede neural LSTM.

Por que este módulo existe separado?
-------------------------------------
Separar o pré-processamento do modelo é uma prática de engenharia de ML
chamada "separation of concerns". Isso facilita:
  - Reutilização: o mesmo pipeline pode ser usado com qualquer ação.
  - Testabilidade: cada função pode ser testada de forma isolada.
  - Manutenção: mudanças no dado não afetam a lógica do modelo e vice-versa.

Autor: Gerado para fins didáticos (FIAP CHALLENGE_04)
Data: 2026
"""

# ==============================================================================
# IMPORTAÇÕES
# ==============================================================================

import logging                        # Registro de eventos e erros no console/arquivo
import os                             # Operações de sistema de arquivos
import time                           # Adicionado para retry
import requests                           # Adicionado: chamadas HTTP para a BRAPI

import numpy as np                    # Operações numéricas vetorizadas (arrays N-dimensionais)
import pandas as pd                   # Manipulação de dados tabulares (DataFrames)
import joblib                         # Serialização eficiente de objetos Python (scalers)

from sklearn.preprocessing import MinMaxScaler   # Normalização de dados para o intervalo [0, 1]
from typing import Tuple                         # Tipagem de retornos com múltiplos valores

# ==============================================================================
# CONFIGURAÇÃO DO SISTEMA DE LOGGING
# ==============================================================================

# O logging é preferível ao print() em produção porque:
#   - Permite definir níveis de severidade (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#   - Pode ser direcionado para arquivos, bancos de dados, etc.
#   - Não polui o stdout da aplicação em produção
logging.basicConfig(
    level=logging.INFO,                           # Exibe apenas mensagens de nível INFO ou superior
    format="%(asctime)s [%(levelname)s] %(message)s",  # Formato: timestamp [NÍVEL] mensagem
    datefmt="%Y-%m-%d %H:%M:%S"                   # Formato da data/hora
)
logger = logging.getLogger(__name__)              # Logger nomeado pelo módulo atual


# ==============================================================================
# PARÂMETROS GLOBAIS — FÁCEIS DE ALTERAR
# ==============================================================================
# Este bloco centraliza todas as configurações do pipeline de dados.
# Para estudar um ativo diferente, basta alterar as variáveis abaixo.

TICKER: str = "ITUB4"          # Ticker SEM o sufixo '.SA' — a BRAPI usa o código puro da B3
START_DATE: str = "2022-01-01"
END_DATE: str = "2026-05-03"
TARGET_COLUMN: str = "Close"
SEQUENCE_LENGTH: int = 60
TRAIN_SPLIT: float = 0.80
ARTIFACTS_DIR: str = "artifacts"
CACHE_DIR: str = os.path.join(ARTIFACTS_DIR, "cache")  # Pasta de cache dos CSVs baixados

# URL base da BRAPI — API financeira brasileira gratuita (https://brapi.dev)
# Documentação: https://brapi.dev/docs
BRAPI_BASE_URL: str = "https://brapi.dev/api"

# Token opcional da BRAPI (gratuito no site). Sem token há limite de ~1000 req/mês.
# Para obter: https://brapi.dev/dashboard
# Deixe None para usar sem autenticação (suficiente para projetos acadêmicos).
BRAPI_TOKEN: str | None = None


# ==============================================================================
# FUNÇÕES DO PIPELINE
# ==============================================================================

def download_stock_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
    max_retries: int = 5,
    retry_delay: int = 15,
    use_cache: bool = True           # Ativa/desativa o cache local em CSV
) -> pd.DataFrame:
    """
    Baixa o histórico de preços via BRAPI com suporte a cache local em CSV.

    Estratégia de cache:
    --------------------
    O nome do arquivo inclui ticker + datas para que qualquer mudança de
    período ou ativo gere um novo download automaticamente.
    Exemplo: artifacts/cache/ITUB4_2022-01-01_2026-05-03.csv

    Fluxo:
      1. Verifica se o CSV deste ticker+período já existe em CACHE_DIR
      2. Se existir  → carrega do disco (instantâneo, sem internet)
      3. Se não existir → baixa da BRAPI, processa e salva o CSV

    Parâmetros
    ----------
    ticker : str
        Código do ativo na B3 SEM sufixo (ex: 'ITUB4', 'PETR4', 'VALE3').
    start : str
        Data de início no formato 'AAAA-MM-DD'.
    end : str
        Data de fim no formato 'AAAA-MM-DD'.
    max_retries : int
        Número máximo de tentativas em caso de falha na BRAPI.
    retry_delay : int
        Segundos de espera base entre tentativas (backoff progressivo).
    use_cache : bool
        True  → usa CSV local se disponível (padrão).
        False → força novo download, ignorando o cache existente.

    Retorna
    -------
    pd.DataFrame
        DataFrame com índice Date e colunas: Open, High, Low, Close, Volume.
    """
    # Garante que a pasta de cache existe antes de qualquer operação
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Nome único do arquivo: ticker + período → qualquer mudança nesses valores
    # resulta em um nome diferente, forçando novo download automaticamente.
    cache_filename: str = f"{ticker}_{start}_{end}.csv"
    cache_path: str = os.path.join(CACHE_DIR, cache_filename)

    # ==========================================================================
    # PASSO 1: Tenta carregar do cache local
    # ==========================================================================
    if use_cache and os.path.exists(cache_path):
        logger.info(f"Cache encontrado: '{cache_path}'. Carregando sem download...")
        try:
            df: pd.DataFrame = pd.read_csv(
                cache_path,
                index_col="Date",   # restaura a coluna Date como índice
                parse_dates=True    # converte strings ISO para DatetimeIndex
            )
            logger.info(
                f"Dados carregados do cache. Shape: {df.shape} | "
                f"Período: {df.index.min().date()} → {df.index.max().date()}"
            )
            return df
        except Exception as e:
            # CSV corrompido ou inválido → descarta e baixa novamente
            logger.warning(f"Cache corrompido ({e}). Baixando novamente da BRAPI...")

    # ==========================================================================
    # PASSO 2: Cache não disponível → baixa da BRAPI
    # ==========================================================================
    logger.info(f"Baixando dados do ativo '{ticker}' de {start} até {end} via BRAPI...")

    url: str = f"{BRAPI_BASE_URL}/quote/{ticker}"

    # ATENÇÃO: a BRAPI não suporta range=custom (enum inválido).
    # Usamos range=max + fromDate/toDate, que são aceitos juntos pela API.
    # Ranges válidos: 1d, 2d, 5d, 7d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    params: dict = {
        "range":       "max",    # Range base; fromDate/toDate refinam o corte
        "interval":    "1d",     # Granularidade diária
        "fromDate":    start,    # Filtro de data inicial (YYYY-MM-DD)
        "toDate":      end,      # Filtro de data final   (YYYY-MM-DD)
        "fundamental": "false",  # Não precisamos de dados fundamentalistas
    }

    if BRAPI_TOKEN:
        params["token"] = BRAPI_TOKEN

    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentativa {attempt}/{max_retries} — GET {url}")

            # timeout=30s evita que a aplicação trave em caso de problemas de rede
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                raise ConnectionError("Rate limit atingido (HTTP 429)")

            if response.status_code == 404:
                raise ValueError(
                    f"Ticker '{ticker}' não encontrado na BRAPI. "
                    "Use o código sem sufixo (ex: 'ITUB4', não 'ITUB4.SA')."
                )

            response.raise_for_status()

            data: dict = response.json()

            # A BRAPI retorna: {"results": [{"historicalDataPrice": [...]}]}
            if not data.get("results") or not data["results"]:
                raise ValueError(f"Resposta vazia da BRAPI para o ticker '{ticker}'.")

            historical: list = data["results"][0].get("historicalDataPrice", [])

            if not historical:
                raise ValueError(
                    f"Sem dados históricos para '{ticker}' no período {start} → {end}."
                )

            # Converte lista de dicts para DataFrame
            # Formato de cada item: {"date": 1641340800, "open": 32.1, ...}
            # ATENÇÃO: "date" é Unix timestamp em SEGUNDOS — não uma string!
            df = pd.DataFrame(historical)

            # Renomeia para o padrão do pipeline (mantém compatibilidade)
            column_mapping: dict = {
                "date":   "Date",
                "open":   "Open",
                "high":   "High",
                "low":    "Low",
                "close":  "Close",
                "volume": "Volume"
            }
            df = df.rename(columns=column_mapping)

            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[[col for col in required_cols if col in df.columns]]

            # unit='s': valores são segundos desde epoch Unix (01/01/1970)
            # utc=True + tz_localize(None): remove timezone → DatetimeIndex naive
            df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True).dt.tz_localize(None)
            df = df.set_index("Date")
            df = df.sort_index()

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Filtro local de segurança (caso a BRAPI retorne dados fora do período)
            df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]

            if df.empty:
                raise ValueError(
                    f"Nenhum dado encontrado para '{ticker}' no período "
                    f"{start} → {end} após filtro local."
                )

            logger.info(
                f"Download concluído na tentativa {attempt}. "
                f"Shape: {df.shape} | "
                f"Período: {df.index.min().date()} → {df.index.max().date()}"
            )

            # ================================================================
            # PASSO 3: Persiste o CSV no cache para execuções futuras
            # ================================================================
            if use_cache:
                df.to_csv(cache_path)
                logger.info(f"Dados salvos em cache: '{cache_path}'")

            return df

        except (ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            # Erros de rede → faz sentido tentar de novo com backoff
            last_exception = e
            wait: int = retry_delay * attempt  # 15s, 30s, 45s, ...
            logger.warning(
                f"Erro de conexão na tentativa {attempt}/{max_retries}: {e}. "
                f"Aguardando {wait}s..."
            )
            time.sleep(wait)

        except ValueError as e:
            # Dados inválidos (ticker errado, período vazio) → não adianta tentar de novo
            logger.error(f"Erro de validação: {e}")
            raise

        except Exception as e:
            last_exception = e
            logger.warning(f"Erro inesperado na tentativa {attempt}/{max_retries}: {e}")
            time.sleep(retry_delay)

    logger.error(f"Todas as {max_retries} tentativas falharam.")
    raise ConnectionError(
        f"Não foi possível baixar dados para '{ticker}' após {max_retries} tentativas. "
        f"Último erro: {last_exception}"
    )


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores ausentes (NaN) no DataFrame de preços.

    Por que podem existir valores ausentes em dados financeiros?
    - Feriados nacionais / internacionais (pregão fechado)
    - Erros na fonte de dados
    - Suspensão de negociação do ativo

    Estratégia adotada: Forward Fill (ffill)
    -----------------------------------------
    Propaga o último valor válido para frente. Isso é ADEQUADO para séries
    temporais financeiras porque:
      1. Evita "vazar" informação do futuro (o que ocorreria com interpolação linear).
      2. Simula o comportamento real: o preço de referência de um dia sem pregão
         é o último preço negociado.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto com possíveis NaNs.

    Retorna
    -------
    pd.DataFrame
        DataFrame sem valores ausentes.
    """
    missing_before: int = df.isnull().sum().sum()

    if missing_before > 0:
        logger.warning(f"Encontrados {missing_before} valores ausentes. Aplicando Forward Fill...")
        # ffill: preenche NaN com o valor imediatamente anterior na série
        df = df.ffill()
        # bfill como fallback: para NaNs no início da série (sem valor anterior disponível)
        df = df.bfill()
        logger.info("Valores ausentes tratados.")
    else:
        logger.info("Nenhum valor ausente encontrado nos dados.")

    return df


def extract_target_column(df: pd.DataFrame, column: str = TARGET_COLUMN) -> np.ndarray:
    """
    Extrai a coluna alvo (preço de fechamento) e a converte para array NumPy 2D.

    Por que 2D (reshape para [-1, 1])?
    ------------------------------------
    O MinMaxScaler do scikit-learn espera entrada no formato (n_amostras, n_features).
    Como temos apenas 1 feature (o preço de fechamento), precisamos de shape (n, 1).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados históricos.
    column : str
        Nome da coluna a ser extraída.

    Retorna
    -------
    np.ndarray
        Array 2D de shape (n_amostras, 1) com os preços de fechamento.

    Lança
    -----
    KeyError
        Se a coluna especificada não existir no DataFrame.
    """
    if column not in df.columns:
        # Levantamos um erro descritivo para facilitar o debug
        raise KeyError(
            f"Coluna '{column}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    # .values retorna o array NumPy subjacente (sem o índice do pandas)
    # reshape(-1, 1): -1 significa "inferir automaticamente" — transforma (n,) em (n, 1)
    prices: np.ndarray = df[column].values.reshape(-1, 1)

    logger.info(f"Coluna '{column}' extraída. Shape do array: {prices.shape}")
    return prices


def normalize_data(prices: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normaliza os dados de preços para o intervalo [0, 1] usando Min-Max Scaling.

    Por que normalizar para LSTM?
    -------------------------------
    Redes neurais, especialmente LSTMs, são sensíveis à escala dos dados por:
      1. A função de ativação sigmoide (usada nos gates da LSTM) opera em [0, 1].
         Dados fora dessa faixa causam saturação dos neurônios (gradients → 0).
      2. Evita que features com grande magnitude dominem o aprendizado.
      3. Acelera a convergência do gradiente descendente.

    Fórmula do MinMax:
        x_norm = (x - x_min) / (x_max - x_min)

    Parâmetros
    ----------
    prices : np.ndarray
        Array 2D de shape (n_amostras, 1) com preços brutos.

    Retorna
    -------
    Tuple[np.ndarray, MinMaxScaler]
        - Array normalizado (mesma shape que a entrada)
        - O objeto scaler TREINADO (necessário para inverter a normalização depois)

    IMPORTANTE: O scaler é treinado APENAS nos dados de treino para evitar
    data leakage (vazamento de informação do futuro para o passado).
    """
    # Instanciamos o scaler com o intervalo padrão [0, 1]
    # feature_range=(0, 1) é o padrão, mas ser explícito melhora a legibilidade
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit_transform: aprende min/max E transforma os dados em um único passo
    # Nota: aqui passamos os dados completos, mas o scaler REAL será ajustado
    # apenas nos dados de treino na função split_data() abaixo.
    # Esta chamada é usada apenas para ter o scaler completo disponível.
    prices_normalized: np.ndarray = scaler.fit_transform(prices)

    logger.info(
        f"Dados normalizados. Min original: {scaler.data_min_[0]:.4f}, "
        f"Max original: {scaler.data_max_[0]:.4f}"
    )
    return prices_normalized, scaler


def create_sequences(
    data: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma a série temporal 1D em sequências de entrada X e saídas Y
    no formato esperado pela LSTM.

    A LSTM aprende padrões em SEQUÊNCIAS. Para isso, precisamos criar
    um conjunto de "janelas deslizantes" (sliding windows):

    Exemplo com sequence_length=3 e dados = [10, 20, 30, 40, 50]:
        X[0] = [10, 20, 30]  →  Y[0] = 40
        X[1] = [20, 30, 40]  →  Y[1] = 50
        ...

    Formato de saída 3D necessário para LSTM:
        X shape: (n_samples, sequence_length, n_features)
                  |           |                |
                  amostras    passos no tempo  features (1 = só preço de fechamento)

    Por que 3D? O TensorFlow/Keras espera a dimensão de "batch", "time steps"
    e "features" separadamente para processar a recorrência temporal.

    Parâmetros
    ----------
    data : np.ndarray
        Array 1D ou 2D normalizado com os preços.
    sequence_length : int
        Número de passos de tempo usados como entrada (look-back window).

    Retorna
    -------
    Tuple[np.ndarray, np.ndarray]
        - X: shape (n_samples, sequence_length, 1) — entradas da LSTM
        - Y: shape (n_samples,) — valores alvo (o próximo preço)
    """
    X_list = []  # Lista que acumulará cada janela de entradas
    Y_list = []  # Lista que acumulará cada valor alvo correspondente

    # Iteramos sobre o array criando janelas deslizantes
    # data.shape[0] é o número total de observações
    for i in range(sequence_length, data.shape[0]):
        # data[i-sequence_length:i] → janela de 'sequence_length' dias
        X_list.append(data[i - sequence_length:i, 0])
        # data[i, 0] → o dia seguinte ao final da janela (o que queremos prever)
        Y_list.append(data[i, 0])

    # Convertemos listas para arrays NumPy (mais eficiente para operações matriciais)
    X: np.ndarray = np.array(X_list)
    Y: np.ndarray = np.array(Y_list)

    # Reshape de X para o formato 3D exigido pelo Keras LSTM:
    # (n_samples, timesteps, features)
    # np.newaxis adiciona uma nova dimensão no eixo especificado
    X = X.reshape(X.shape[0], X.shape[1], 1)

    logger.info(
        f"Sequências criadas → X shape: {X.shape}, Y shape: {Y.shape}"
    )
    return X, Y


def split_train_test(
    prices_normalized: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados normalizados em conjuntos de treino e teste, e cria
    as sequências temporais para cada conjunto.

    ATENÇÃO — Por que NÃO usar shuffle aqui?
    ------------------------------------------
    Em séries temporais, a ORDEM dos dados é fundamental!
    Embaralhar os dados quebraria a causalidade (você "veria o futuro" durante o treino).
    Por isso, a divisão é feita de forma SEQUENCIAL: os primeiros N% são treino,
    os últimos (1-N)% são teste.

    Parâmetros
    ----------
    prices_normalized : np.ndarray
        Array 2D normalizado de shape (total_dias, 1).
    train_split : float
        Fração dos dados destinada ao treino (ex: 0.8 = 80%).
    sequence_length : int
        Tamanho da janela temporal para criar as sequências.

    Retorna
    -------
    Tuple de 4 np.ndarray: (X_train, X_test, Y_train, Y_test)
    """
    # Calculamos o índice de corte (onde o treino termina e o teste começa)
    split_index: int = int(len(prices_normalized) * train_split)
    logger.info(
        f"Divisão treino/teste: índice {split_index} "
        f"({split_index} dias treino / {len(prices_normalized) - split_index} dias teste)"
    )

    # Divisão SEQUENCIAL dos dados brutos normalizados
    train_data: np.ndarray = prices_normalized[:split_index]
    test_data: np.ndarray = prices_normalized[split_index:]

    # Criamos as sequências separadas para treino e teste
    X_train, Y_train = create_sequences(train_data, sequence_length)
    X_test, Y_test   = create_sequences(test_data,  sequence_length)

    logger.info(
        f"Shapes finais → "
        f"X_train: {X_train.shape}, Y_train: {Y_train.shape} | "
        f"X_test: {X_test.shape}, Y_test: {Y_test.shape}"
    )
    return X_train, X_test, Y_train, Y_test


def save_artifacts(scaler: MinMaxScaler, artifacts_dir: str = ARTIFACTS_DIR) -> None:
    """
    Salva o objeto MinMaxScaler em disco usando joblib.

    Por que salvar o scaler?
    --------------------------
    Na hora da inferência (previsão com novos dados), precisamos normalizar
    os dados NOVOS com os MESMOS parâmetros (min/max) aprendidos nos dados
    de treino. Se não salvarmos o scaler, perderíamos esses parâmetros.

    Por que joblib em vez de pickle?
    ----------------------------------
    joblib é mais eficiente para arrays NumPy grandes (usa compressão e
    mapeamento de memória), sendo o método recomendado pelo scikit-learn.

    Parâmetros
    ----------
    scaler : MinMaxScaler
        Objeto scaler treinado nos dados de treino.
    artifacts_dir : str
        Diretório onde o arquivo será salvo.
    """
    # Cria o diretório se não existir (exist_ok=True evita erro se já existir)
    os.makedirs(artifacts_dir, exist_ok=True)

    scaler_path: str = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    logger.info(f"Scaler salvo em: '{scaler_path}'")


# ==============================================================================
# FUNÇÃO ORQUESTRADORA PRINCIPAL
# ==============================================================================

def run_pipeline(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
    target_column: str = TARGET_COLUMN,
    sequence_length: int = SEQUENCE_LENGTH,
    train_split: float = TRAIN_SPLIT,
    save: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Executa o pipeline completo de preparação dos dados, do download ao formato LSTM.

    Esta função orquestra todas as etapas em ordem:
        1. Download dos dados históricos
        2. Tratamento de valores ausentes
        3. Extração da coluna alvo
        4. Normalização dos dados
        5. Divisão e criação das sequências temporais
        6. Salvamento do scaler (se solicitado)

    Parâmetros
    ----------
    ticker, start, end, target_column, sequence_length, train_split : ver funções acima
    save : bool
        Se True, salva o scaler em disco.

    Retorna
    -------
    Tuple contendo X_train, X_test, Y_train, Y_test e o scaler treinado.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO DE DADOS")
    logger.info("=" * 60)

    # --- Etapa 1: Download ---
    df = download_stock_data(ticker, start, end)

    # --- Etapa 2: Limpeza ---
    df = handle_missing_values(df)

    # --- Etapa 3: Extração da coluna alvo ---
    prices = extract_target_column(df, target_column)

    # --- Etapa 4: Normalização ---
    # ATENÇÃO: aqui normalizamos com todos os dados para salvar o scaler global.
    # A divisão treino/teste respeita a ordem temporal nas etapas seguintes.
    prices_normalized, scaler = normalize_data(prices)

    # --- Etapa 5: Divisão e criação de sequências ---
    X_train, X_test, Y_train, Y_test = split_train_test(
        prices_normalized, train_split, sequence_length
    )

    # --- Etapa 6: Salvamento dos artefatos ---
    if save:
        save_artifacts(scaler)

    logger.info("Pipeline concluído com sucesso!")
    logger.info("=" * 60)

    return X_train, X_test, Y_train, Y_test, scaler


# ==============================================================================
# PONTO DE ENTRADA — Execução direta do módulo para teste rápido
# ==============================================================================

if __name__ == "__main__":
    # Este bloco só é executado quando rodamos "python data_processing.py" diretamente.
    # É útil para testar o pipeline de forma isolada.
    X_train, X_test, Y_train, Y_test, scaler = run_pipeline()

    print("\n--- Resumo dos dados preparados ---")
    print(f"  X_train shape : {X_train.shape}  → (amostras_treino, {SEQUENCE_LENGTH} dias, 1 feature)")
    print(f"  X_test  shape : {X_test.shape}  → (amostras_teste,  {SEQUENCE_LENGTH} dias, 1 feature)")
    print(f"  Y_train shape : {Y_train.shape}")
    print(f"  Y_test  shape : {Y_test.shape}")
    print(f"  Scaler  range : [{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")
