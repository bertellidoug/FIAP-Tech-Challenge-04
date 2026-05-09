"""
app.py
======
API RESTful desenvolvida com FastAPI para servir previsões do modelo LSTM
de preços de ações em produção.

Por que FastAPI?
-----------------
FastAPI é o framework Python mais adequado para APIs de ML em produção:
  1. Performance: baseado em Starlette + uvicorn (ASGI), próximo ao desempenho
     de Go/Node.js para I/O concorrente.
  2. Tipagem: usa type hints nativos do Python para validação automática de
     payloads com Pydantic — zero código boilerplate de validação.
  3. Documentação automática: gera Swagger UI (/docs) e ReDoc (/redoc)
     automaticamente a partir dos type hints e docstrings.
  4. Assíncrono: suporte nativo a async/await para operações I/O intensivas.

Arquitetura da API:
-------------------
  POST /predict          → Recebe preços históricos e retorna previsão
  GET  /health           → Health check (liveness probe para orquestrador)
  GET  /model/info       → Metadados do modelo em produção

Middleware de monitoramento:
-----------------------------------
  Um middleware registra o tempo de resposta de CADA requisição, o endpoint
  acessado e o status HTTP. Isso é essencial para:
    - Detectar degradação de performance
    - Alertas de SLA (ex: P95 > 500ms)
    - Auditoria e rastreabilidade em sistemas financeiros

Autor: Gerado para fins didáticos (FIAP CHALLENGE_04)
Data: 2026
"""

# ==============================================================================
# IMPORTAÇÕES
# ==============================================================================

import logging
import time
import uuid                           # Geração de IDs únicos por requisição (request tracing)
from contextlib import asynccontextmanager  # Gerenciador de ciclo de vida do FastAPI
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

# FastAPI e utilitários de HTTP
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware   # Permite chamadas cross-origin (ex: frontend)
from fastapi.responses import JSONResponse

# Pydantic: validação de dados de entrada e saída com type hints
# Versão 2 (incluída no FastAPI moderno) usa model_config em vez de class Config
from pydantic import BaseModel, Field, field_validator

# Módulo de exportação — carrega modelo e scaler do disco
from export import load_all_artifacts, load_metadata, SCALER_PATH, MODEL_PATH
from data_processing import SEQUENCE_LENGTH


# ==============================================================================
# CONFIGURAÇÃO DO LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# ESTADO GLOBAL DA APLICAÇÃO
# ==============================================================================
# Armazenar modelo e scaler em variáveis globais é a abordagem correta para APIs
# de ML: carregamos UMA VEZ na inicialização e reutilizamos em todas as requisições.
# Não usar global state para dados de USUÁRIO (não é thread-safe para isso).

class AppState:
    """
    Contêiner para o estado compartilhado da aplicação.
    Centraliza os artefatos carregados para serem acessíveis pelos endpoints.
    """
    model: tf.keras.Model = None      # Modelo LSTM carregado
    scaler = None                      # MinMaxScaler carregado
    metadata: Dict[str, Any] = {}     # Metadados do último treino


app_state = AppState()


# ==============================================================================
# CICLO DE VIDA DA APLICAÇÃO (Lifespan)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação: startup e shutdown.

    O decorator @asynccontextmanager divide o código em duas fases:
      - Antes do 'yield': código de INICIALIZAÇÃO (carrega modelo, conecta BD, etc.)
      - Após o 'yield':   código de ENCERRAMENTO (fecha conexões, libera recursos)

    Por que usar lifespan em vez de @app.on_event("startup")?
    ----------------------------------------------------------
    O @app.on_event foi deprecado no FastAPI 0.93+. O lifespan é o padrão
    moderno e garante que tanto startup quanto shutdown estejam no mesmo
    contexto de código (mais fácil de entender e manter).
    """
    # ---- STARTUP ----
    logger.info("=" * 60)
    logger.info("Inicializando API LSTM Stock Predictor...")
    logger.info("=" * 60)

    try:
        # Carrega modelo e scaler do disco (operação pesada — apenas uma vez)
        app_state.model, app_state.scaler = load_all_artifacts(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH
        )
        # Carrega metadados para o endpoint /model/info
        app_state.metadata = load_metadata()

        logger.info("API pronta para receber requisições.")

    except FileNotFoundError as e:
        # Se o modelo não existir, logamos o erro mas não crashamos o server.
        # O endpoint /predict retornará 503 (Service Unavailable) neste caso.
        logger.critical(f"FALHA NO STARTUP: {e}")
        logger.critical("A API está rodando mas PREVISÕES estarão indisponíveis.")
        logger.critical("Execute 'python model.py' para treinar e salvar o modelo.")

    yield  # <-- A API fica rodando aqui. Após 'yield', é o shutdown.

    # ---- SHUTDOWN ----
    logger.info("Encerrando API. Liberando recursos...")
    # Em uma aplicação real, fecharíamos conexões de banco de dados,
    # pools de conexão, clientes de monitoramento, etc.
    app_state.model  = None
    app_state.scaler = None
    logger.info("API encerrada com sucesso.")


# ==============================================================================
# INSTÂNCIA DA APLICAÇÃO FASTAPI
# ==============================================================================

app = FastAPI(
    title="LSTM Stock Price Predictor API",
    description=(
        "API RESTful para previsão de preços de ações utilizando rede neural LSTM. "
        "Treinada com dados históricos do Yahoo Finance. "
        "Desenvolvida para o FIAP CHALLENGE_04."
    ),
    version="1.0.0",
    lifespan=lifespan,  # Registra o gerenciador de ciclo de vida
    docs_url="/docs",   # Swagger UI (documentação interativa)
    redoc_url="/redoc"  # ReDoc (documentação alternativa mais elegante)
)


# ==============================================================================
# MIDDLEWARE DE CORS
# ==============================================================================

# CORS (Cross-Origin Resource Sharing): permite que aplicações web em outros
# domínios (ex: frontend em React em localhost:3000) chamem esta API.
# Em produção, substitua allow_origins=["*"] pelo domínio específico do seu frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Permitir todos os origens (ajustar em produção!)
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],        # Authorization, Content-Type, etc.
)


# ==============================================================================
# MIDDLEWARE DE MONITORAMENTO — Logging de tempo de resposta
# ==============================================================================

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """
    Middleware que intercepta TODAS as requisições HTTP para:
      1. Gerar um ID único por requisição (Request ID) — essencial para
         rastrear logs de uma mesma requisição em sistemas distribuídos.
      2. Medir e registrar o tempo de processamento (latência).
      3. Registrar método HTTP, endpoint, status code e latência.

    O padrão "request_id" permite correlacionar logs de uma mesma transação
    em diferentes serviços (API Gateway, banco de dados, modelo, etc.).

    Como funciona o middleware no FastAPI?
    ----------------------------------------
    O middleware envolve cada requisição:
        requisição → middleware → endpoint → middleware → resposta

    'call_next(request)' executa o endpoint e retorna a resposta.
    O código antes é executado na IDA, o código depois na VOLTA.
    """
    # Gera um ID único para rastrear esta requisição (UUID4 = aleatório seguro)
    request_id: str = str(uuid.uuid4())[:8]  # Usamos apenas os primeiros 8 chars para brevidade

    # Registra o momento de entrada antes de processar o endpoint
    start_time: float = time.perf_counter()  # perf_counter é mais preciso que time.time()

    # Adiciona o request_id ao estado da requisição para uso nos endpoints
    request.state.request_id = request_id

    logger.info(
        f"[{request_id}] → {request.method} {request.url.path} | "
        f"Cliente: {request.client.host if request.client else 'desconhecido'}"
    )

    # Chama o próximo handler na cadeia (o endpoint real)
    response = await call_next(request)

    # Calcula a latência em milissegundos
    elapsed_ms: float = (time.perf_counter() - start_time) * 1000

    # Adiciona header customizado na resposta para que o cliente também possa ver a latência
    # Isso é útil para debugging via browser DevTools ou Postman
    response.headers["X-Request-ID"]      = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"

    # Define o nível de log com base na latência (alerta se > 1 segundo)
    log_level = logging.WARNING if elapsed_ms > 1000 else logging.INFO

    logger.log(
        log_level,
        f"[{request_id}] ← {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Tempo: {elapsed_ms:.2f}ms"
        + (" ⚠ LATÊNCIA ALTA!" if elapsed_ms > 1000 else "")
    )

    return response


# ==============================================================================
# SCHEMAS DE ENTRADA E SAÍDA (Pydantic)
# ==============================================================================

class PredictRequest(BaseModel):
    """
    Schema de entrada para o endpoint POST /predict.

    O que é Pydantic?
    ------------------
    Pydantic valida automaticamente os dados recebidos no body da requisição
    contra este schema. Se os dados não baterem, o FastAPI retorna HTTP 422
    (Unprocessable Entity) com detalhes do erro — sem precisar escrever
    nenhuma lógica de validação manual.

    Campos
    ------
    prices : List[float]
        Lista de preços históricos de fechamento (mínimo: SEQUENCE_LENGTH itens).
        O modelo usará os últimos SEQUENCE_LENGTH valores para fazer a previsão.
    steps : int
        Número de passos futuros a prever (default: 1 = próximo dia).
    """
    prices: List[float] = Field(
        ...,                          # '...' = campo obrigatório (sem default)
        min_length=SEQUENCE_LENGTH,   # Validação automática: mínimo de itens
        description=(
            f"Lista de preços históricos de fechamento. "
            f"Mínimo de {SEQUENCE_LENGTH} valores necessários."
        ),
        examples=[[28.5, 29.1, 28.8, 29.5]]   # Exemplos para o Swagger UI
    )
    steps: int = Field(
        default=1,
        ge=1,              # greater or equal: steps >= 1
        le=30,             # less or equal: steps <= 30 (proteção contra previsões excessivas)
        description="Número de passos futuros a prever (1 a 30 dias)."
    )

    @field_validator("prices")
    @classmethod
    def validate_prices_positive(cls, v: List[float]) -> List[float]:
        """
        Validador customizado: todos os preços devem ser positivos.
        Preços negativos são inválidos para ações.
        """
        if any(p <= 0 for p in v):
            raise ValueError("Todos os preços devem ser maiores que zero.")
        return v


class PredictResponse(BaseModel):
    """
    Schema de saída do endpoint POST /predict.

    Documentar schemas de saída é boa prática: gera documentação automática
    e permite que o frontend saiba exatamente o que esperar.

    Campos
    ------
    predictions : List[float]
        Lista com os preços previstos para os próximos 'steps' dias.
    steps : int
        Número de passos previstos.
    model_version : str
        Versão do modelo usado (rastreabilidade).
    request_id : str
        ID único desta requisição (para suporte/debugging).
    """
    predictions: List[float] = Field(
        ...,
        description="Preços previstos para os próximos N dias (na escala original)."
    )
    steps: int = Field(..., description="Número de passos previstos.")
    model_version: str = Field(..., description="Versão/identificador do modelo.")
    request_id: str = Field(..., description="ID único desta requisição.")


class HealthResponse(BaseModel):
    """Schema de saída do endpoint GET /health."""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    message: str


# ==============================================================================
# ENDPOINTS DA API
# ==============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Verifica se a API e o modelo estão operacionais. Usado por orquestradores como Kubernetes."
)
async def health_check() -> HealthResponse:
    """
    Endpoint de verificação de saúde (liveness/readiness probe).

    Por que ter um endpoint de health check?
    -----------------------------------------
    Em produção, orquestradores como Kubernetes enviam requisições periódicas
    para este endpoint para decidir se o container está saudável.
    Se retornar erro (5xx), o orquestrador reinicia o container automaticamente.

    Liveness  probe: "o container está vivo?" → GET /health
    Readiness probe: "o container está pronto para receber tráfego?" → GET /health
    """
    model_loaded:  bool = app_state.model  is not None
    scaler_loaded: bool = app_state.scaler is not None
    is_healthy:    bool = model_loaded and scaler_loaded

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=model_loaded,
        scaler_loaded=scaler_loaded,
        message=(
            "Todos os componentes operacionais."
            if is_healthy
            else "ATENÇÃO: Modelo ou scaler não carregados. Execute 'python model.py'."
        )
    )


@app.get(
    "/model/info",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Informações do Modelo",
    description="Retorna metadados do modelo em produção: métricas, configurações, data do treino."
)
async def model_info() -> Dict[str, Any]:
    """
    Retorna os metadados do modelo carregado em produção.

    Útil para monitorar degradação de performance ao longo do tempo:
    se as métricas do modelo atual estiverem ruins, é hora de retreinar.
    """
    if not app_state.metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadados do modelo não disponíveis. Execute o treinamento primeiro."
        )
    return app_state.metadata


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Prever Preço de Fechamento",
    description=(
        f"Recebe uma lista de preços históricos de fechamento (mínimo {SEQUENCE_LENGTH} valores) "
        "e retorna a previsão do(s) próximo(s) dia(s) usando o modelo LSTM treinado."
    )
)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """
    Endpoint principal de previsão.

    Fluxo de processamento:
    ------------------------
    1. Valida que modelo e scaler estão carregados
    2. Pega os últimos SEQUENCE_LENGTH preços da lista enviada
    3. Normaliza os dados com o mesmo scaler do treino
    4. Formata entrada no shape 3D esperado pela LSTM: (1, SEQUENCE_LENGTH, 1)
    5. Gera previsão(ões) iterativamente (multi-step forecasting)
    6. Inverte a normalização para obter preços na escala real
    7. Retorna o resultado com rastreabilidade

    Multi-step forecasting:
    -----------------------
    Para prever mais de 1 passo à frente, usamos a estratégia "recursive":
    a previsão do passo N é adicionada à janela para gerar o passo N+1.
    Esta abordagem acumula erro a cada passo, por isso limitamos 'steps' a 30.
    """
    # --- Verificação de disponibilidade do modelo ---
    if app_state.model is None or app_state.scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modelo não carregado. Verifique se o arquivo "
                f"'{MODEL_PATH}' existe e execute 'python model.py' se necessário."
            )
        )

    request_id: str = getattr(request.state, "request_id", "N/A")
    logger.info(
        f"[{request_id}] Previsão solicitada: "
        f"{len(body.prices)} preços fornecidos, {body.steps} passo(s) à frente."
    )

    try:
        # --- Etapa 1: Preparação dos dados de entrada ---
        # Pegamos os últimos SEQUENCE_LENGTH preços (a LSTM só usa esta janela)
        # Isso permite que o usuário envie mais do que o necessário sem erro.
        recent_prices: np.ndarray = np.array(body.prices[-SEQUENCE_LENGTH:], dtype=np.float32)

        # Reshape para (n, 1) — formato exigido pelo MinMaxScaler
        prices_2d: np.ndarray = recent_prices.reshape(-1, 1)

        # --- Etapa 2: Normalização com o scaler do treino ---
        # CRÍTICO: usamos transform() (NÃO fit_transform()) nos dados novos.
        # fit_transform() aprenderia um novo min/max, quebrando a consistência.
        prices_normalized: np.ndarray = app_state.scaler.transform(prices_2d)

        # Mantemos uma janela deslizante para previsão multi-step
        # Iniciamos com os dados normalizados de entrada
        current_sequence: np.ndarray = prices_normalized.flatten().tolist()

        predictions_normalized: List[float] = []

        # --- Etapa 3: Previsão iterativa (recursive multi-step) ---
        for step in range(body.steps):
            # Pega os últimos SEQUENCE_LENGTH valores como janela de entrada
            input_window: np.ndarray = np.array(
                current_sequence[-SEQUENCE_LENGTH:]
            ).reshape(1, SEQUENCE_LENGTH, 1)  # Shape: (1 batch, 60 timesteps, 1 feature)

            # Inferência: model.predict() retorna shape (1, 1)
            # [0][0] extrai o valor escalar
            next_pred_normalized: float = float(app_state.model.predict(input_window, verbose=0)[0][0])
            predictions_normalized.append(next_pred_normalized)

            # Adiciona a previsão à janela para o próximo passo (estratégia recursive)
            current_sequence.append(next_pred_normalized)

        # --- Etapa 4: Inversão da normalização ---
        # Transformamos o array de previsões de volta para a escala real (preços em moeda)
        preds_array: np.ndarray = np.array(predictions_normalized).reshape(-1, 1)
        predictions_real: List[float] = (
            app_state.scaler.inverse_transform(preds_array)
            .flatten()
            .tolist()
        )

        # Arredondamos para 4 casas decimais (precisão adequada para preços de ações)
        predictions_real = [round(p, 4) for p in predictions_real]

        logger.info(
            f"[{request_id}] Previsões geradas: {predictions_real}"
        )

        return PredictResponse(
            predictions=predictions_real,
            steps=body.steps,
            model_version=app_state.metadata.get("saved_at", "N/A"),
            request_id=request_id
        )

    except Exception as e:
        # Capturamos qualquer exceção inesperada para não expor detalhes internos ao cliente
        # Em produção, erros internos devem ser logados detalhadamente mas retornados de forma
        # genérica ao cliente (princípio de não expor detalhes de implementação)
        logger.error(f"[{request_id}] Erro durante a previsão: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar a previsão. Consulte os logs. Request ID: {request_id}"
        )


# ==============================================================================
# PONTO DE ENTRADA — Execução direta com uvicorn
# ==============================================================================

if __name__ == "__main__":
    """
    Permite rodar a API diretamente via:
        python app.py

    Em produção, prefira iniciar com o comando uvicorn diretamente:
        uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

    Por que uvicorn?
    ----------------
    uvicorn é um servidor ASGI (Asynchronous Server Gateway Interface) de alta
    performance baseado em uvloop (reimplementação de asyncio em Cython/C).
    É o servidor de produção recomendado para FastAPI.

    workers=4: usa 4 processos paralelos para suportar mais requisições simultâneas.
    """
    import uvicorn

    uvicorn.run(
        "app:app",         # "módulo:instância_do_app"
        host="0.0.0.0",    # Escuta em todas as interfaces de rede (necessário no Docker)
        port=8000,          # Porta padrão da aplicação
        reload=False,       # True apenas em desenvolvimento (reinicia ao salvar o arquivo)
        log_level="info",   # Nível de log do uvicorn
        access_log=True     # Registra todas as requisições de acesso
    )
