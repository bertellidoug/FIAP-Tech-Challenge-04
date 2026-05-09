"""
model.py
========
Módulo responsável por:
  1. Definir a arquitetura da rede neural LSTM
  2. Compilar e treinar o modelo com as melhores práticas
  3. Avaliar o modelo com métricas financeiras (MAE, RMSE, MAPE)
  4. Exportar o modelo treinado para uso posterior pela API

O que é uma LSTM?
-----------------
LSTM (Long Short-Term Memory) é um tipo especial de Rede Neural Recorrente (RNN)
projetada para aprender dependências de longo prazo em dados sequenciais.

O problema das RNNs simples é o "Vanishing Gradient": ao propagar o erro
para trás no tempo (BPTT), os gradientes tendem a zero e o modelo "esquece"
eventos distantes.

A LSTM resolve isso com uma "célula de memória" e 3 portões (gates):
  - Forget Gate (portão de esquecimento): decide O QUE esquecer da memória
  - Input Gate  (portão de entrada):      decide O QUE novo adicionar à memória
  - Output Gate (portão de saída):        decide O QUE da memória enviar como saída

Isso permite que o modelo aprenda padrões de curto E longo prazo na série temporal.

Autor: Gerado para fins didáticos (FIAP CHALLENGE_04)
Data: 2026
"""

# ==============================================================================
# IMPORTAÇÕES
# ==============================================================================

import logging
import os
import time       # Para medir o tempo de treinamento

import numpy as np
import joblib

# TensorFlow / Keras — framework de deep learning do Google
# Keras é a API de alto nível que facilita a construção de redes neurais
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model   # type: ignore
from tensorflow.keras.layers import (                        # type: ignore
    LSTM,         # Camada recorrente principal (Long Short-Term Memory)
    Dense,        # Camada totalmente conectada (fully connected / linear)
    Dropout,      # Técnica de regularização para prevenir overfitting
    Input         # Camada de entrada explícita (boa prática em Keras funcional)
)
from tensorflow.keras.callbacks import (                     # type: ignore
    EarlyStopping,   # Para treino quando a melhora estagna (evita overfitting)
    ModelCheckpoint, # Salva automaticamente o melhor modelo durante o treino
    ReduceLROnPlateau # Reduz a taxa de aprendizado quando o val_loss estagna
)
from tensorflow.keras.optimizers import Adam                 # type: ignore

# Métricas de avaliação do scikit-learn — padrão da indústria para regressão
from sklearn.metrics import mean_absolute_error, mean_squared_error

from typing import Dict, Tuple

# Importamos nosso módulo de pré-processamento
from data_processing import (
    run_pipeline,
    ARTIFACTS_DIR,
    SEQUENCE_LENGTH,
    TICKER,
    START_DATE,
    END_DATE
)

# Importamos a função orquestradora de exportação de artefatos
# save_all_artifacts salva: modelo (.keras) + scaler (.pkl) + metadados (.json)
from export import save_all_artifacts

# ==============================================================================
# CONFIGURAÇÃO DO LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# HIPERPARÂMETROS DO MODELO — Centralizados para fácil ajuste
# ==============================================================================
# Hiperparâmetros são configurações do modelo que NÃO são aprendidas durante
# o treino. Eles precisam ser definidos antes e ajustados manualmente (ou via
# técnicas como GridSearch/Optuna).

# --- Arquitetura ---
LSTM_UNITS_1: int = 128    # Neurônios na 1ª camada LSTM. Mais unidades = mais capacidade,
                            # mas maior risco de overfitting e custo computacional

LSTM_UNITS_2: int = 64     # Neurônios na 2ª camada LSTM (menor = afunilamento)

DENSE_UNITS: int = 32      # Neurônios na camada densa intermediária

DROPOUT_RATE: float = 0.20 # Taxa de Dropout: 20% dos neurônios são "desligados"
                            # aleatoriamente a cada passo. Força o modelo a aprender
                            # representações redundantes → menos overfitting.

# --- Treinamento ---
LEARNING_RATE: float = 1e-3  # Taxa de aprendizado inicial do otimizador Adam.
                               # Controla o "tamanho do passo" na descida do gradiente.
                               # Muito alto → oscila, não converge. Muito baixo → lento.

BATCH_SIZE: int = 32           # Número de amostras processadas antes de atualizar os pesos.
                                # Valores comuns: 16, 32, 64. Menor = gradiente mais ruidoso
                                # mas generaliza melhor. Maior = mais estável, mas mais memória.

EPOCHS: int = 100              # Máximo de épocas. EarlyStopping vai parar antes se necessário.
PATIENCE: int = 15             # Quantas épocas sem melhora antes de parar (EarlyStopping)

# --- Arquivos de saída ---
MODEL_PATH: str = os.path.join(ARTIFACTS_DIR, "lstm_model.keras")  # Formato nativo do Keras
BEST_MODEL_CHECKPOINT: str = os.path.join(ARTIFACTS_DIR, "best_model_checkpoint.keras")


# ==============================================================================
# CONSTRUÇÃO DA ARQUITETURA DA REDE NEURAL
# ==============================================================================

def build_model(sequence_length: int = SEQUENCE_LENGTH) -> tf.keras.Model:
    """
    Constrói e compila a arquitetura da rede neural LSTM.

    Arquitetura escolhida — Por quê?
    ---------------------------------
    A arquitetura "stacked LSTM" (LSTM empilhado) com Dropout é amplamente
    usada em previsão de séries temporais financeiras porque:

      1. Múltiplas camadas LSTM: camadas inferiores aprendem padrões de curto
         prazo (ex: tendências de 3-5 dias), camadas superiores aprendem
         padrões de longo prazo (ex: sazonalidade mensal).

      2. Dropout: previne co-adaptação dos neurônios. Com 20%, o modelo não
         pode depender de nenhum neurônio específico, sendo forçado a
         distribuir o aprendizado.

      3. Camada Dense final: reduz as representações aprendidas para uma única
         saída escalar (o preço previsto).

    Fluxo de dados:
        Input (batch, 60, 1)
            → LSTM(128, return_sequences=True)  → (batch, 60, 128)
            → Dropout(0.2)
            → LSTM(64, return_sequences=False)  → (batch, 64)
            → Dropout(0.2)
            → Dense(32, activation='relu')       → (batch, 32)
            → Dense(1)                            → (batch, 1) ← previsão

    Parâmetros
    ----------
    sequence_length : int
        Número de passos de tempo na sequência de entrada.

    Retorna
    -------
    tf.keras.Model
        Modelo compilado e pronto para treino.
    """
    logger.info("Construindo arquitetura da rede LSTM...")

    # Sequential: modelo linear onde cada camada passa para a próxima.
    # Adequado aqui pois nossa arquitetura não tem ramificações.
    model = Sequential(name="LSTM_StockPredictor")

    # --- Camada de Entrada ---
    # Explicitar a shape da entrada é boa prática: evita bugs silenciosos
    # shape=(sequence_length, 1): 'sequence_length' passos, cada um com 1 feature
    model.add(Input(shape=(sequence_length, 1), name="input_layer"))

    # --- 1ª Camada LSTM ---
    # return_sequences=True: retorna saída para CADA passo de tempo,
    # não apenas o último. NECESSÁRIO quando há outra camada LSTM após esta,
    # pois a próxima LSTM também precisa de uma sequência como entrada.
    model.add(LSTM(
        units=LSTM_UNITS_1,      # 128 unidades (células de memória)
        return_sequences=True,   # Passa sequência inteira para a próxima camada
        name="lstm_layer_1"
    ))

    # Dropout após a 1ª LSTM para regularização
    # Durante o treino, 20% dos outputs são zerados aleatoriamente
    # Durante a inferência, o Dropout é automaticamente desativado pelo Keras
    model.add(Dropout(rate=DROPOUT_RATE, name="dropout_1"))

    # --- 2ª Camada LSTM ---
    # return_sequences=False (padrão): retorna APENAS a saída do último passo de tempo.
    # Faz sentido aqui porque a camada Dense seguinte precisa de um vetor fixo,
    # não de uma sequência.
    model.add(LSTM(
        units=LSTM_UNITS_2,      # 64 unidades — afunilamento gradual é boa prática
        return_sequences=False,  # Apenas o último estado oculto é passado
        name="lstm_layer_2"
    ))
    model.add(Dropout(rate=DROPOUT_RATE, name="dropout_2"))

    # --- Camada Densa Intermediária ---
    # ReLU (Rectified Linear Unit): f(x) = max(0, x)
    # É a função de ativação mais comum em camadas ocultas porque:
    #   - Não sofre de vanishing gradient (para valores positivos, gradiente = 1)
    #   - Computacionalmente simples
    #   - Introduz não-linearidade (sem ela, a rede seria apenas uma regressão linear)
    model.add(Dense(units=DENSE_UNITS, activation="relu", name="dense_hidden"))

    # --- Camada de Saída ---
    # 1 neurônio, sem ativação (linear por padrão) → adequado para regressão
    # Queremos prever um valor contínuo (o preço), não uma probabilidade.
    # Usar sigmoid/softmax aqui seria um erro pois limitaria a saída a [0,1].
    model.add(Dense(units=1, activation="linear", name="output_layer"))

    # --- Compilação ---
    # Otimizador Adam (Adaptive Moment Estimation):
    #   - Adapta automaticamente a taxa de aprendizado por parâmetro
    #   - Combina RMSProp (gradientes ao quadrado) + SGD com momentum
    #   - Padrão recomendado para a maioria das tarefas
    #
    # Loss: MSE (Mean Squared Error) para regressão — penaliza erros grandes
    # ao quadrado, incentivando o modelo a evitar grandes desvios.
    #
    # Metrics: MAE acompanhado durante o treino para visualização mais intuitiva
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    # Exibe resumo da arquitetura: camadas, shapes e quantidade de parâmetros
    model.summary(print_fn=logger.info)

    return model


# ==============================================================================
# CALLBACKS DE TREINAMENTO
# ==============================================================================

def get_callbacks() -> list:
    """
    Retorna a lista de callbacks usados durante o treinamento.

    Callbacks são funções executadas em pontos específicos do treinamento
    (fim de época, fim de batch, etc.). São essenciais para treinos robustos.

    Retorna
    -------
    list
        Lista de objetos Callback do Keras.
    """
    # --- Early Stopping ---
    # Monitora a perda no conjunto de validação (val_loss)
    # Se não houver melhora por 'patience' épocas, o treino é interrompido.
    # restore_best_weights=True: ao parar, os pesos são restaurados para a
    # época de melhor val_loss (não os da última época, que pode ser pior).
    early_stopping = EarlyStopping(
        monitor="val_loss",       # Métrica a ser monitorada
        patience=PATIENCE,        # Épocas de tolerância sem melhora
        restore_best_weights=True, # Restaura pesos da melhor época
        verbose=1                  # Exibe mensagem quando parar
    )

    # --- Model Checkpoint ---
    # Salva automaticamente o modelo sempre que val_loss melhora.
    # Garante que não percamos o melhor modelo mesmo se o treino for interrompido.
    model_checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_CHECKPOINT,
        monitor="val_loss",
        save_best_only=True,  # Salva apenas se for o melhor até agora
        verbose=1
    )

    # --- Reduce LR on Plateau ---
    # Reduz a taxa de aprendizado quando o val_loss estagna.
    # Ideia: quando o modelo "para de aprender" com o LR atual, um LR menor
    # permite que ele refine os pesos em uma região mais estreita do gradiente.
    # factor=0.5: novo_lr = lr_atual * 0.5 (reduz à metade)
    # min_lr: evita que o LR caia abaixo de um valor mínimo útil
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,          # Fator de redução do LR
        patience=7,           # Épocas sem melhora antes de reduzir
        min_lr=1e-6,          # LR mínimo absoluto
        verbose=1
    )

    return [early_stopping, model_checkpoint, reduce_lr]


# ==============================================================================
# TREINAMENTO DO MODELO
# ==============================================================================

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray
) -> tf.keras.callbacks.History:
    """
    Treina o modelo LSTM com os dados preparados.

    Por que usar validation_data (e não validation_split)?
    -------------------------------------------------------
    Em séries temporais, a ordem importa. Se usássemos validation_split,
    o Keras pegaria as últimas N% amostras do treino para validação, o que
    pode não ser representativo do teste final. Passamos explicitamente
    X_test/Y_test como validação para monitorar a generalização real.

    Parâmetros
    ----------
    model : tf.keras.Model
        Modelo compilado (saída de build_model).
    X_train, Y_train : np.ndarray
        Dados de treino.
    X_test, Y_test : np.ndarray
        Dados de teste/validação.

    Retorna
    -------
    tf.keras.callbacks.History
        Objeto com histórico de perda e métricas por época.
        Útil para plotar curvas de aprendizado.
    """
    logger.info("Iniciando treinamento do modelo LSTM...")
    logger.info(
        f"Configuração: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, "
        f"LR={LEARNING_RATE}, PATIENCE={PATIENCE}"
    )

    # Registramos o tempo de início para medir a duração do treino
    start_time: float = time.time()

    # model.fit() é o método central de treinamento do Keras
    history = model.fit(
        X_train, Y_train,                  # Dados de treino
        epochs=EPOCHS,                      # Número máximo de passagens completas nos dados
        batch_size=BATCH_SIZE,              # Amostras por atualização de pesos
        validation_data=(X_test, Y_test),   # Dados para monitorar overfitting
        callbacks=get_callbacks(),           # Funções auxiliares de treino
        shuffle=False,                       # CRUCIAL: NÃO embaralhar séries temporais!
        verbose=1                            # Exibe barra de progresso por época
    )

    elapsed: float = time.time() - start_time
    logger.info(f"Treinamento concluído em {elapsed:.1f} segundos.")

    return history


# ==============================================================================
# AVALIAÇÃO DO MODELO COM MÉTRICAS FINANCEIRAS
# ==============================================================================

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    scaler          # MinMaxScaler (sem tipagem explícita para evitar import circular)
) -> Dict[str, float]:
    """
    Avalia o modelo nas métricas financeiras padrão: MAE, RMSE e MAPE.

    ATENÇÃO: Inversão da normalização
    -----------------------------------
    As previsões do modelo e os valores reais estão na escala NORMALIZADA [0,1].
    Para calcular métricas em escala REAL (preço em R$ ou USD), precisamos
    desfazer a transformação MinMax usando o mesmo scaler salvo no treino.

    Fórmulas das métricas:
    ----------------------
    MAE  (Mean Absolute Error):
        MAE = (1/n) * Σ|y_real - y_pred|
        Interpretação: erro médio absoluto em unidade original (ex: R$).
        Vantagem: fácil de interpretar. Desvantagem: não penaliza erros grandes.

    RMSE (Root Mean Squared Error):
        RMSE = √[(1/n) * Σ(y_real - y_pred)²]
        Interpretação: similar ao MAE, mas penaliza desproporcionalmente
        erros grandes (por causa do quadrado). Sensível a outliers.

    MAPE (Mean Absolute Percentage Error):
        MAPE = (100/n) * Σ|(y_real - y_pred) / y_real|
        Interpretação: erro percentual médio. Ex: MAPE=2.5% significa que
        o modelo erra, em média, 2.5% do valor real.
        Vantagem: independente de escala (comparável entre diferentes ativos).
        Cuidado: explodir quando y_real ≈ 0.

    Parâmetros
    ----------
    model : tf.keras.Model
        Modelo treinado.
    X_test : np.ndarray
        Entradas do conjunto de teste, shape (n, seq_len, 1).
    Y_test : np.ndarray
        Valores reais do conjunto de teste, shape (n,).
    scaler : MinMaxScaler
        Scaler treinado nos dados originais.

    Retorna
    -------
    Dict[str, float]
        Dicionário com as métricas calculadas.
    """
    logger.info("Avaliando modelo no conjunto de teste...")

    # 1. Gerar previsões na escala normalizada
    # predict() retorna shape (n, 1) — precisamos achatar para (n,)
    Y_pred_normalized: np.ndarray = model.predict(X_test, verbose=0)

    # 2. Inverter a normalização para obter preços reais
    # inverse_transform espera shape (n, 1), por isso usamos reshape
    Y_pred_real: np.ndarray = scaler.inverse_transform(
        Y_pred_normalized.reshape(-1, 1)
    ).flatten()  # .flatten() transforma (n,1) em (n,) para facilitar cálculos

    Y_test_real: np.ndarray = scaler.inverse_transform(
        Y_test.reshape(-1, 1)
    ).flatten()

    # 3. Calcular métricas na escala real (preços em moeda)
    mae: float = mean_absolute_error(Y_test_real, Y_pred_real)

    # RMSE é a raiz quadrada do MSE (mean_squared_error não tem RMSE direto no sklearn)
    rmse: float = float(np.sqrt(mean_squared_error(Y_test_real, Y_pred_real)))

    # MAPE: adicionamos epsilon (1e-10) no denominador para evitar divisão por zero
    mape: float = float(
        np.mean(np.abs((Y_test_real - Y_pred_real) / (Y_test_real + 1e-10))) * 100
    )

    metrics: Dict[str, float] = {
        "MAE":  round(mae,  4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    }

    logger.info("=" * 40)
    logger.info("MÉTRICAS DE AVALIAÇÃO NO CONJUNTO DE TESTE")
    logger.info(f"  MAE  (Erro Médio Absoluto):          {mae:.4f}  (unidades monetárias)")
    logger.info(f"  RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}  (unidades monetárias)")
    logger.info(f"  MAPE (Erro Percentual Médio Absoluto): {mape:.4f}%")
    logger.info("=" * 40)

    return metrics


# ==============================================================================
# EXPORTAÇÃO DO MODELO E ARTEFATOS
# ==============================================================================

def save_model(model: tf.keras.Model, model_path: str = MODEL_PATH) -> None:
    """
    Salva o modelo Keras completo em disco no formato nativo .keras.

    Por que o formato .keras?
    --------------------------
    O formato .keras (introduzido no Keras 3) é o mais robusto e recomendado:
      - Salva: arquitetura + pesos + estado do otimizador + configurações
      - Portável: pode ser carregado em qualquer ambiente com TensorFlow/Keras
      - Seguro: evita vulnerabilidades do formato legado .h5 com pickle

    Parâmetros
    ----------
    model : tf.keras.Model
        Modelo treinado.
    model_path : str
        Caminho do arquivo de destino (ex: "artifacts/lstm_model.keras").
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Modelo salvo em: '{model_path}'")


def load_trained_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Carrega um modelo Keras salvo do disco.

    Usado pela API (app.py) para carregar o modelo sem precisar treinar novamente.

    Parâmetros
    ----------
    model_path : str
        Caminho do arquivo .keras salvo.

    Retorna
    -------
    tf.keras.Model
        Modelo carregado e pronto para inferência.

    Lança
    -----
    FileNotFoundError
        Se o arquivo do modelo não for encontrado.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{model_path}'. "
            "Execute o treinamento primeiro (python model.py)."
        )

    logger.info(f"Carregando modelo de: '{model_path}'")
    model = load_model(model_path)
    logger.info("Modelo carregado com sucesso.")
    return model


# ==============================================================================
# FUNÇÃO PRINCIPAL — Orquestra todo o pipeline de ML
# ==============================================================================

def run_training_pipeline() -> Dict[str, float]:
    """
    Executa o pipeline completo de Machine Learning:
        1. Coleta e pré-processamento dos dados
        2. Construção da arquitetura LSTM
        3. Treinamento com callbacks
        4. Avaliação com métricas
        5. Salvamento do modelo e artefatos

    Retorna
    -------
    Dict[str, float]
        Dicionário com as métricas de avaliação finais.
    """
    logger.info("=" * 60)
    logger.info(f"PIPELINE DE TREINAMENTO — Ativo: {TICKER}")
    logger.info("=" * 60)

    # --- Etapa 1: Dados ---
    # Reutilizamos o pipeline de pré-processamento do módulo data_processing.py
    # O scaler é necessário para inverter a normalização nas métricas
    X_train, X_test, Y_train, Y_test, scaler = run_pipeline(save=True)

    # --- Etapa 2: Modelo ---
    # sequence_length é inferido do shape de X_train (dimensão 1)
    model = build_model(sequence_length=X_train.shape[1])

    # --- Etapa 3: Treinamento ---
    history = train_model(model, X_train, Y_train, X_test, Y_test)

    # --- Etapa 4: Avaliação ---
    metrics = evaluate_model(model, X_test, Y_test, scaler)

    # --- Etapa 5: Exportação ---
    # Chamamos save_all_artifacts (de export.py) em vez de save_model() isolado.
    # Isso garante que TODOS os artefatos sejam persistidos de forma atômica:
    #   - lstm_model.keras          → modelo completo
    #   - scaler.pkl                → MinMaxScaler (necessário para inverter normalização)
    #   - training_metadata.json    → métricas + hiperparâmetros + ticker + datas
    save_all_artifacts(
        model=model,
        scaler=scaler,
        metrics=metrics,
        extra_metadata={
            "ticker":      TICKER,
            "start_date":  START_DATE,
            "end_date":    END_DATE,
            "sequence_length": SEQUENCE_LENGTH,
            "epochs_max":  EPOCHS,
            "batch_size":  BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lstm_units_1": LSTM_UNITS_1,
            "lstm_units_2": LSTM_UNITS_2,
            "dropout_rate": DROPOUT_RATE,
        }
    )

    logger.info("Pipeline de treinamento concluído com sucesso!")
    logger.info(f"  → Modelo salvo em:         {MODEL_PATH}")
    logger.info(f"  → Scaler salvo em:         {os.path.join(ARTIFACTS_DIR, 'scaler.pkl')}")
    logger.info(f"  → Metadados salvos em:     {os.path.join(ARTIFACTS_DIR, 'training_metadata.json')}")
    logger.info(f"  → Métricas finais:         {metrics}")

    return metrics


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # Reprodutibilidade: fixar seeds garante que o treinamento produz os
    # mesmos resultados em execuções diferentes (importante para debugging
    # e comparação de experimentos).
    RANDOM_SEED: int = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    logger.info(f"Seeds fixadas para reprodutibilidade: {RANDOM_SEED}")

    # Verificar se há GPU disponível (treino muito mais rápido com GPU)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU(s) detectada(s): {[g.name for g in gpus]} — Usando aceleração GPU!")
    else:
        logger.info("Nenhuma GPU detectada. Treinando na CPU (pode ser mais lento).")

    # Executa o pipeline completo
    final_metrics = run_training_pipeline()

    print("\n" + "=" * 50)
    print("RESULTADOS FINAIS DO MODELO")
    print("=" * 50)
    for metric_name, metric_value in final_metrics.items():
        unit = "%" if metric_name == "MAPE" else ""
        print(f"  {metric_name:5s}: {metric_value:.4f}{unit}")
    print("=" * 50)
    print(f"\nArtefatos salvos no diretório: '{ARTIFACTS_DIR}/'")
