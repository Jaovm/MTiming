# collect_fundamental_data.py
import requests
import pandas as pd
import os
import json
import time

# --- Configurações ---
API_TOKEN = "5gVedSQ928pxhFuTvBFPfr" # Token fornecido pelo usuário
BASE_URL = "https://brapi.dev/api"
DATA_DIR = "/home/ubuntu/data/fundamental"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Lista de Tickers (sem .SA)
TICKERS = [
    "AGRO3", "BBAS3", "BBSE3", "BPAC11", "EGIE3",
    "ITUB3", "PRIO3", "PSSA3", "SAPR3", "SBSP3",
    "VIVT3", "WEGE3", "TOTS3", "B3SA3", "TAEE3",
    "CMIG3"
]

# Módulos a serem buscados: Nenhum específico, apenas fundamental=true
# A API indicou que o plano só permite 'summaryProfile', que deve vir com fundamental=true
# MODULES = "summaryProfile" # Removido para testar se fundamental=true basta

# Cria o diretório se não existir
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_brapi_fundamental_data(ticker):
    """Busca dados fundamentalistas ATUAIS de um ticker na API brapi.dev."""
    endpoint = f"/quote/{ticker}"
    params = {
        # "modules": MODULES, # Removido
        "fundamental": "true" # Solicita dados fundamentais (atuais)
        # Token é passado via Header
    }
    url = BASE_URL + endpoint
    print(f"Buscando dados fundamentalistas ATUAIS para {ticker}...")
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=60)
        response.raise_for_status() # Lança exceção para erros HTTP
        data = response.json()

        # Verifica se há resultados e se não há erro na resposta
        if not data or not data.get("results") or data.get("error"): 
            error_msg = data.get("error", "Resposta vazia ou sem resultados.")
            print(f"Erro ao buscar dados para {ticker}: {error_msg}")
            return None

        # Extrai os dados do summaryProfile (esperado dentro de results)
        results = data["results"][0] # Pega o primeiro resultado
        
        # Verifica se o summaryProfile está presente (pode não vir para todos os ativos)
        if not results.get("summaryProfile") and not results.get("marketCap"):
             print(f"Dados fundamentais (summaryProfile ou marketCap) não encontrados para {ticker} na resposta.")
             # Ainda assim, salva o que veio para inspeção
             # return None # Decide se quer salvar mesmo sem dados fundamentais

        # Simplificando: salvamos todo o 'results' que contém os dados atuais
        fundamental_data = results

        return fundamental_data

    except requests.exceptions.Timeout:
        print(f"Timeout ao buscar dados para {ticker}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão ao buscar dados para {ticker}: {e}")
        # Imprime o conteúdo da resposta se for um erro do cliente (4xx)
        if e.response is not None and 400 <= e.response.status_code < 500:
            try:
                print(f"Detalhe do erro ({ticker}): {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Detalhe do erro ({ticker}): {e.response.text}")
        return None
    except Exception as e:
        print(f"Erro inesperado ao processar dados para {ticker}: {e}")
        return None

# Coleta e salva os dados para cada ticker
all_fundamental_data = {}
print(f"Iniciando coleta de dados ATUAIS para {len(TICKERS)} tickers...")
for ticker in TICKERS:
    data = fetch_brapi_fundamental_data(ticker)
    if data:
        all_fundamental_data[ticker] = data
        # Salvar dados brutos em JSON para análise posterior
        # Usar um nome diferente para indicar que são dados atuais/snapshot
        file_path = os.path.join(DATA_DIR, f"{ticker}_fundamental_snapshot.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Dados snapshot de {ticker} salvos em {file_path}")
        except Exception as e:
            print(f"Erro ao salvar JSON para {ticker}: {e}")
    else:
        print(f"Não foi possível obter dados snapshot para {ticker}.")

    # Pausa para evitar limites de taxa
    time.sleep(1.5) # Pausa um pouco menor, pois a resposta deve ser mais leve

print(f"\nColeta de dados snapshot concluída. {len(all_fundamental_data)}/{len(TICKERS)} tickers coletados com sucesso.")

# Próximo passo: Usar esses dados snapshot no painel Streamlit

