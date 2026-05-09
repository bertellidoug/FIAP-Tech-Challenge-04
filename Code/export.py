"""
export.py
=========
Módulo responsável por centralizar toda a lógica de persistência (salvar e
carregar) dos artefatos de ML gerados pelo treinamento:

  - Modelo LSTM treinado  (.keras)
  - Scaler de normalização (.pkl via joblib)
  - Metadados da execução  (.json) — ex: métricas, data do treino, ticker

Por que um módulo dedicado de exportação?
-----------------------------------------
Seguindo o princípio de responsabilidade única (SRP - Single Responsibility
Principle), este módulo abstrai COMO os artefatos são salvos/carregados.
Se no futuro quisermos trocar joblib por pickle, ou salvar na nuvem (S3, GCS),
a mudança ocorre APENAS aqui, sem impactar model.py ou app.py.

Isso também facilita versionamento de modelos: é fácil adicionar um prefixo
com timestamp ao nome dos arquivos para manter histórico de treinamentos.

Autor: Gerado para fins didáticos (FIAP CHALLENGE_04)
Data: 2026
"""

# ==============================================================================
# IMPORTAÇÕES
# ==============================================================================

import json       # Serialização de dicionários Python para formato JSON (texto)
import logging
import os
from datetime import datetime  # Para registrar data/hora do treino nos metadados
from typing import Any, Dict, Optional, Tuple

import joblib     # Serialização eficiente para objetos NumPy/scikit-learn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model   # type: ignore
from sklearn.preprocessing import MinMaxScaler

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
# CONSTANTES DE CAMINHOS — centralizadas aqui para fácil manutenção
# ==============================================================================

ARTIFACTS_DIR: str = "artifacts"

# Nomes dos arquivos de artefatos
MODEL_FILENAME: str   = "lstm_model.keras"      # Modelo completo (arquitetura + pesos)
SCALER_FILENAME: str  = "scaler.pkl"            # MinMaxScaler serializado
METADATA_FILENAME: str = "training_metadata.json"  # Metadados do treino em JSON legível

# Caminhos completos (junção do diretório com o nome do arquivo)
MODEL_PATH: str    = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
SCALER_PATH: str   = os.path.join(ARTIFACTS_DIR, SCALER_FILENAME)
METADATA_PATH: str = os.path.join(ARTIFACTS_DIR, METADATA_FILENAME)


# ==============================================================================
# FUNÇÕES DE SALVAMENTO
# ==============================================================================

def ensure_artifacts_dir(artifacts_dir: str = ARTIFACTS_DIR) -> None:
    """
    Garante que o diretório de artefatos existe.

    Abstrai a criação de diretório para ser reutilizado por todas as funções
    de salvamento, evitando código duplicado (princípio DRY: Don't Repeat Yourself).

    Parâmetros
    ----------
    artifacts_dir : str
        Caminho do diretório a ser criado se não existir.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    logger.debug(f"Diretório de artefatos garantido: '{artifacts_dir}'")


def save_model(
    model: tf.keras.Model,
    path: str = MODEL_PATH
) -> str:
    """
    Persiste o modelo Keras completo no formato nativo .keras.

    O que é salvo no arquivo .keras?
    -----------------------------------
    1. Arquitetura completa (JSON interno da rede)
    2. Pesos treinados (arrays NumPy das matrizes de pesos e biases)
    3. Estado do otimizador (permite retomar o treino exatamente de onde parou)
    4. Configuração de compilação (loss, métricas, etc.)

    Por que .keras e não .h5?
    --------------------------
    O .h5 é o formato legado baseado em HDF5. O formato .keras (nativo do
    Keras 3+) é mais seguro (sem uso de pickle interno), mais rápido e
    totalmente suportado. O .h5 ainda funciona mas não é mais recomendado.

    Parâmetros
    ----------
    model : tf.keras.Model
        Modelo treinado a ser salvo.
    path : str
        Caminho de destino do arquivo.

    Retorna
    -------
    str
        Caminho absoluto onde o modelo foi salvo (útil para logging/API).
    """
    ensure_artifacts_dir(os.path.dirname(path))
    model.save(path)
    abs_path: str = os.path.abspath(path)
    logger.info(f"✓ Modelo LSTM salvo em: '{abs_path}'")
    return abs_path


def save_scaler(
    scaler: MinMaxScaler,
    path: str = SCALER_PATH
) -> str:
    """
    Persiste o objeto MinMaxScaler em disco usando joblib.

    Por que precisamos salvar o scaler?
    -------------------------------------
    O scaler foi "treinado" (fit) nos dados históricos de treino e aprendeu
    os valores mínimo e máximo da série de preços. Para normalizar NOVOS dados
    de entrada na API, precisamos dos MESMOS parâmetros. Sem o scaler salvo:
      1. Não conseguimos normalizar entradas novas corretamente.
      2. Não conseguimos reverter as previsões para a escala de preço real.

    Por que joblib e não pickle?
    -----------------------------
    joblib usa um protocolo otimizado para arrays NumPy grandes:
      - Mapeamento de memória (memory-mapped files): carregamento sem copiar dados
      - Compressão nativa opcional
      - Recomendado oficialmente pelo scikit-learn para seus objetos

    Parâmetros
    ----------
    scaler : MinMaxScaler
        Scaler ajustado nos dados de treino.
    path : str
        Caminho de destino do arquivo.

    Retorna
    -------
    str
        Caminho absoluto onde o scaler foi salvo.
    """
    ensure_artifacts_dir(os.path.dirname(path))
    joblib.dump(scaler, path)
    abs_path: str = os.path.abspath(path)
    logger.info(f"✓ Scaler salvo em: '{abs_path}'")
    return abs_path


def save_metadata(
    metadata: Dict[str, Any],
    path: str = METADATA_PATH
) -> str:
    """
    Salva metadados do treinamento em um arquivo JSON legível por humanos.

    Por que salvar metadados?
    --------------------------
    Metadados documentam QUANDO e COM QUE CONFIGURAÇÃO o modelo foi treinado.
    São essenciais para:
      - Rastreabilidade (Model Tracking): saber qual versão do modelo está em produção
      - Reprodutibilidade: recriar as mesmas condições de treino
      - Monitoramento: verificar se as métricas degradaram em novos treinos
      - Auditoria: em aplicações financeiras, rastreabilidade é CRÍTICA

    Formato JSON é escolhido por ser:
      - Legível por humanos (sem precisar de código para visualizar)
      - Facilmente consumível por sistemas de CI/CD, dashboards MLOps, etc.
      - Nativo ao ecossistema web (compatível com a API)

    Parâmetros
    ----------
    metadata : Dict[str, Any]
        Dicionário com métricas, hiperparâmetros, ticker, datas, etc.
    path : str
        Caminho de destino do arquivo JSON.

    Retorna
    -------
    str
        Caminho absoluto onde o JSON foi salvo.
    """
    ensure_artifacts_dir(os.path.dirname(path))

    # Adicionamos a data/hora de salvamento automaticamente
    # isoformat() gera string no padrão ISO 8601: "2026-05-02T14:30:00.123456"
    metadata["saved_at"] = datetime.now().isoformat()

    # indent=4: formata o JSON com indentação de 4 espaços (legível por humanos)
    # ensure_ascii=False: permite caracteres UTF-8 (ex: acentos em textos)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    abs_path: str = os.path.abspath(path)
    logger.info(f"✓ Metadados salvos em: '{abs_path}'")
    return abs_path


def save_all_artifacts(
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    metrics: Dict[str, float],
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Função orquestradora que salva TODOS os artefatos de uma vez.

    Convenção: esta é a função que deve ser chamada ao final do treinamento.
    Ela agrupa todos os salvamentos e retorna os caminhos para referência.

    Parâmetros
    ----------
    model : tf.keras.Model
        Modelo LSTM treinado.
    scaler : MinMaxScaler
        Scaler ajustado nos dados de treino.
    metrics : Dict[str, float]
        Métricas de avaliação calculadas (MAE, RMSE, MAPE).
    extra_metadata : Dict[str, Any], opcional
        Informações adicionais a incluir nos metadados (ex: ticker, datas).

    Retorna
    -------
    Dict[str, str]
        Dicionário com os caminhos absolutos dos artefatos salvos.
    """
    logger.info("Salvando todos os artefatos do treinamento...")

    # Salva modelo e scaler
    model_path  = save_model(model)
    scaler_path = save_scaler(scaler)

    # Monta o dicionário de metadados enriquecido
    metadata: Dict[str, Any] = {
        "model_path":  os.path.abspath(MODEL_PATH),
        "scaler_path": os.path.abspath(SCALER_PATH),
        "metrics":     metrics,
    }
    # Mescla com metadados extras passados como argumento (se houver)
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = save_metadata(metadata)

    paths: Dict[str, str] = {
        "model":    model_path,
        "scaler":   scaler_path,
        "metadata": metadata_path
    }

    logger.info("Todos os artefatos salvos com sucesso:")
    for artifact_name, artifact_path in paths.items():
        logger.info(f"  [{artifact_name}] → {artifact_path}")

    return paths


# ==============================================================================
# FUNÇÕES DE CARREGAMENTO
# ==============================================================================

def load_model_from_disk(path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Carrega o modelo Keras salvo do disco.

    Esta função é usada pela API (app.py) na inicialização para carregar
    o modelo uma única vez e mantê-lo em memória para todas as requisições.

    Carregar o modelo uma vez na inicialização (não a cada requisição) é
    fundamental para performance: modelos LSTM podem ter dezenas de MB e
    o carregamento levaria centenas de milissegundos por requisição.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo .keras.

    Retorna
    -------
    tf.keras.Model
        Modelo carregado, compilado e pronto para inferência.

    Lança
    -----
    FileNotFoundError
        Se o arquivo não for encontrado — sinaliza que o treino é necessário.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{path}'.\n"
            "Execute o treinamento antes de iniciar a API:\n"
            "  python model.py"
        )

    logger.info(f"Carregando modelo de: '{path}'...")
    model = load_model(path)
    logger.info("Modelo carregado com sucesso.")
    return model


def load_scaler_from_disk(path: str = SCALER_PATH) -> MinMaxScaler:
    """
    Carrega o MinMaxScaler salvo do disco via joblib.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo .pkl.

    Retorna
    -------
    MinMaxScaler
        Scaler com os parâmetros min/max aprendidos nos dados de treino.

    Lança
    -----
    FileNotFoundError
        Se o arquivo não for encontrado.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler não encontrado em '{path}'.\n"
            "Execute o treinamento antes de iniciar a API:\n"
            "  python model.py"
        )

    logger.info(f"Carregando scaler de: '{path}'...")
    scaler: MinMaxScaler = joblib.load(path)
    logger.info(
        f"Scaler carregado. "
        f"Range original: [{scaler.data_min_[0]:.4f}, {scaler.data_max_[0]:.4f}]"
    )
    return scaler


def load_metadata(path: str = METADATA_PATH) -> Dict[str, Any]:
    """
    Carrega os metadados do último treinamento do arquivo JSON.

    Útil para endpoint de status da API (ex: GET /model/info) que retorna
    informações sobre o modelo ativo em produção.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo JSON de metadados.

    Retorna
    -------
    Dict[str, Any]
        Metadados do último treinamento, ou dicionário vazio se não existir.
    """
    if not os.path.exists(path):
        logger.warning(f"Arquivo de metadados não encontrado em '{path}'. Retornando vazio.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        metadata: Dict[str, Any] = json.load(f)

    logger.info(f"Metadados carregados de: '{path}'")
    return metadata


def load_all_artifacts(
    model_path: str  = MODEL_PATH,
    scaler_path: str = SCALER_PATH
) -> Tuple[tf.keras.Model, MinMaxScaler]:
    """
    Carrega modelo e scaler de uma vez — função de conveniência para a API.

    Parâmetros
    ----------
    model_path : str
        Caminho do modelo .keras.
    scaler_path : str
        Caminho do scaler .pkl.

    Retorna
    -------
    Tuple[tf.keras.Model, MinMaxScaler]
        Tupla (modelo, scaler) prontos para inferência.
    """
    model  = load_model_from_disk(model_path)
    scaler = load_scaler_from_disk(scaler_path)
    logger.info("Artefatos carregados e prontos para uso.")
    return model, scaler


# ==============================================================================
# PONTO DE ENTRADA — Teste de roundtrip (salvar + carregar)
# ==============================================================================

if __name__ == "__main__":
    """
    Executa um teste simples de 'roundtrip':
      1. Verifica se os artefatos existem
      2. Carrega modelo e scaler
      3. Imprime informações de verificação

    Útil para confirmar que o salvamento funcionou corretamente após o treino.
    """
    logger.info("Executando teste de carregamento de artefatos...")

    try:
        model, scaler = load_all_artifacts()
        metadata = load_metadata()

        print("\n--- Verificação dos Artefatos ---")
        print(f"  Modelo:     {model.name} | Parâmetros: {model.count_params():,}")
        print(f"  Scaler:     min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f}")
        print(f"  Metadados:  {json.dumps(metadata, indent=4, ensure_ascii=False)}")
        print("\n✓ Todos os artefatos carregados com sucesso!")

    except FileNotFoundError as e:
        print(f"\n✗ Erro: {e}")
