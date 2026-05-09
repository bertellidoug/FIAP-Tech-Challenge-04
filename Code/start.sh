#!/usr/bin/env bash
# ==============================================================================
# start.sh
# ==============================================================================
# Script de inicialização da aplicação para deploy em plataformas como
# Render, Heroku, Railway ou qualquer servidor Linux/Unix.
#
# Como usar:
#   1. Dê permissão de execução (apenas na primeira vez):
#        chmod +x start.sh
#
#   2. Execute o script:
#        ./start.sh
#
# O que este script faz, em sequência:
#   1. Ativa o ambiente virtual Python (se existir)
#   2. Instala/atualiza as dependências do requirements.txt
#   3. Verifica se os artefatos do modelo já existem
#   4. Se não existirem, executa o treinamento automaticamente
#   5. Inicia o servidor uvicorn (API FastAPI)
# ==============================================================================

# 'set -e': encerra o script imediatamente se qualquer comando retornar erro.
# Sem isso, o script continuaria mesmo após uma falha, causando erros silenciosos.
set -e

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DE CORES PARA OUTPUT LEGÍVEL NO LOG
# ------------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color — reseta a cor

log_info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${BLUE}======================================${NC}"; \
                echo -e "${BLUE}  $1${NC}"; \
                echo -e "${BLUE}======================================${NC}"; }

# ------------------------------------------------------------------------------
# 2. DIRETÓRIO DE TRABALHO
# Garante que o script sempre roda a partir do diretório onde ele está localizado,
# independente de onde foi chamado.
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
log_info "Diretório de trabalho: $SCRIPT_DIR"

# ------------------------------------------------------------------------------
# 3. AMBIENTE VIRTUAL PYTHON
# Ativa o virtualenv se existir. Em plataformas PaaS, o ambiente geralmente
# já está ativo, mas verificamos para compatibilidade local.
# ------------------------------------------------------------------------------
log_section "Ambiente Python"

VENV_PATH="$SCRIPT_DIR/.venv"

if [ -d "$VENV_PATH" ]; then
    log_info "Ativando ambiente virtual: $VENV_PATH"
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    log_info "Python ativo: $(which python)"
    log_info "Versão Python: $(python --version)"
else
    log_warn "Ambiente virtual não encontrado em '$VENV_PATH'."
    log_warn "Usando Python do sistema: $(which python3 || which python)"
fi

# ------------------------------------------------------------------------------
# 4. INSTALAÇÃO DAS DEPENDÊNCIAS
# Instala/atualiza todas as bibliotecas listadas no requirements.txt.
# '--no-cache-dir' evita uso de cache desatualizado.
# '--quiet' reduz o output verboso do pip (mas mantém erros visíveis).
# ------------------------------------------------------------------------------
log_section "Instalando Dependências"

if [ -f "requirements.txt" ]; then
    log_info "Instalando dependências de requirements.txt..."
    pip install --no-cache-dir --quiet -r requirements.txt
    log_info "Dependências instaladas com sucesso."
else
    log_error "Arquivo requirements.txt não encontrado!"
    exit 1
fi

# ------------------------------------------------------------------------------
# 5. VERIFICAÇÃO E TREINAMENTO DO MODELO
# Se os artefatos ainda não existem (primeiro deploy ou reset), treina o modelo.
# Em deploys subsequentes, o modelo já existe e etapa é pulada.
#
# IMPORTANTE: em produção real, o modelo deve ser treinado OFFLINE e os
# artefatos versionados junto ao código ou em um bucket de storage (S3, GCS).
# Treinar no startup apenas é aceitável para demos/MVPs.
# ------------------------------------------------------------------------------
log_section "Verificando Artefatos do Modelo"

ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
MODEL_FILE="$ARTIFACTS_DIR/lstm_model.keras"
SCALER_FILE="$ARTIFACTS_DIR/scaler.pkl"

if [ -f "$MODEL_FILE" ] && [ -f "$SCALER_FILE" ]; then
    log_info "Artefatos encontrados. Pulando treinamento."
    log_info "  Modelo : $MODEL_FILE"
    log_info "  Scaler : $SCALER_FILE"
else
    log_warn "Artefatos NÃO encontrados. Iniciando treinamento do modelo..."
    log_warn "Isso pode levar vários minutos dependendo do hardware disponível."

    if [ -f "model.py" ]; then
        python model.py

        # Verifica se o treinamento gerou os artefatos esperados
        if [ -f "$MODEL_FILE" ] && [ -f "$SCALER_FILE" ]; then
            log_info "Treinamento concluído. Artefatos gerados com sucesso."
        else
            log_error "Treinamento falhou: artefatos não foram criados."
            log_error "Verifique os logs acima para detalhes do erro."
            exit 1
        fi
    else
        log_error "Arquivo model.py não encontrado. Impossível treinar o modelo."
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 6. CONFIGURAÇÃO DA PORTA
# A variável PORT pode ser definida pela plataforma (Render, Heroku) ou
# manualmente antes de chamar o script. Default: 8000.
# ------------------------------------------------------------------------------
log_section "Iniciando Servidor"

APP_PORT="${PORT:-8000}"     # Usa $PORT se definida, senão usa 8000 como padrão
APP_WORKERS="${WORKERS:-2}"  # Permite configurar workers via variável de ambiente

log_info "Configuração do servidor:"
log_info "  Host    : 0.0.0.0"
log_info "  Porta   : $APP_PORT"
log_info "  Workers : $APP_WORKERS"
log_info "  Módulo  : app:app"

# ------------------------------------------------------------------------------
# 7. INICIALIZAÇÃO DO SERVIDOR UVICORN
# 'exec' substitui o processo do shell pelo uvicorn.
# Isso é importante para que sinais do sistema operacional (SIGTERM, SIGINT)
# sejam enviados diretamente ao uvicorn, permitindo shutdown gracioso.
# Sem 'exec', o shell receberia o sinal e o uvicorn poderia ser encerrado
# abruptamente sem concluir requisições em andamento.
# ------------------------------------------------------------------------------
log_info "Iniciando uvicorn... Acesse a API em http://0.0.0.0:${APP_PORT}/docs"

exec uvicorn app:app \
    --host 0.0.0.0 \
    --port "$APP_PORT" \
    --workers "$APP_WORKERS" \
    --log-level info \
    --access-log
