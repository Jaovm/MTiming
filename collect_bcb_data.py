import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append("/opt/.manus/.sandbox-runtime")

# Define o período de coleta (últimos 10 anos)
today = datetime.now()
start_date = today - timedelta(days=10*365)
start_date_str = start_date.strftime("%d/%m/%Y")
end_date_str = today.strftime("%d/%m/%Y")

# Códigos das séries no SGS do BCB
series_codes = {
    "IPCA": 433,       # Índice Nacional de Preços ao Consumidor Amplo
    "IGP-M": 189,      # Índice Geral de Preços - Mercado
    "Selic_Meta": 1178, # Taxa de juros - Selic - Meta Selic definida pelo Copom
    "Selic_Efetiva": 11, # Taxa de juros - Selic
    "IBC-Br": 24368,    # Índice de Atividade Econômica do Banco Central (IBC-Br) - Dessazonalizado
    "Desemprego_PNAD": 24369 # Taxa de desocupação - PNAD Contínua - IBGE (trimestral, usar último disponível)
}

# Diretório para salvar os dados
data_dir = "/home/ubuntu/data"

# Cria o diretório se não existir (usando shell)
import os
os.makedirs(data_dir, exist_ok=True)

def fetch_bcb_sgs_data(series_code, start_date, end_date):
    """Busca dados de uma série do SGS do BCB."""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status() # Lança exceção para erros HTTP
        data = response.json()
        if not data: # Verifica se a lista está vazia
            print(f"Nenhum dado retornado para a série {series_code} no período solicitado.")
            # Tenta buscar os últimos 100 dados se o período falhar (útil para séries menos frequentes)
            url_last = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados/ultimos/100?formato=json"
            print(f"Tentando buscar os últimos 100 dados para a série {series_code}...")
            response = requests.get(url_last, timeout=60)
            response.raise_for_status()
            data = response.json()
            if not data:
                print(f"Nenhum dado retornado para a série {series_code} (nem últimos 100).")
                return None

        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.set_index("data")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar dados para a série {series_code}: {e}")
        return None
    except Exception as e:
        print(f"Erro ao processar dados para a série {series_code}: {e}")
        return None

# Coleta e salva os dados para cada série
for name, code in series_codes.items():
    print(f"Coletando dados para {name} (Série {code})...")
    df_series = fetch_bcb_sgs_data(code, start_date_str, end_date_str)
    if df_series is not None and not df_series.empty:
        file_path = os.path.join(data_dir, f"{name.lower()}_data.csv")
        df_series.to_csv(file_path)
        print(f"Dados de {name} salvos em {file_path}")
    else:
        print(f"Não foi possível salvar os dados para {name}.")

print("\nColeta de dados do BCB (atualizada) concluída.")

