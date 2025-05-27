import sys
sys.path.append("/opt/.manus/.sandbox-runtime")
from data_api import ApiClient
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# Initialize API Client
client = ApiClient()

# Define symbols and parameters
symbols = {
    "IBOV": "^BVSP",
    "VIX": "^VIX" # Embora o VIX seja americano, é um indicador de risco global relevante
}

# Define o período de coleta (últimos 10 anos)
# Yahoo Finance API usa 'range' ou 'period1'/'period2'
# Usaremos 'range' para simplicidade
data_range = "10y"
interval = "1d" # Intervalo diário

# Diretório para salvar os dados
data_dir = "/home/ubuntu/data"
os.makedirs(data_dir, exist_ok=True)

def fetch_yahoo_finance_data(symbol, region, data_range, interval):
    """Busca dados históricos de um símbolo no Yahoo Finance."""
    try:
        print(f"Buscando dados para {symbol} (Região: {region}, Range: {data_range}, Intervalo: {interval})")
        # Ajusta a região para o Ibovespa
        api_region = "BR" if symbol == "^BVSP" else "US"
        stock_data = client.call_api(
            "YahooFinance/get_stock_chart",
            query={
                "symbol": symbol,
                "region": api_region,
                "range": data_range,
                "interval": interval,
                "includeAdjustedClose": True
            }
        )

        # Verifica se houve erro na resposta da API
        if stock_data.get("chart", {}).get("error"):
            print(f"Erro da API ao buscar {symbol}: {stock_data['chart']['error']}")
            return None

        # Verifica se os resultados existem
        result = stock_data.get("chart", {}).get("result", [])
        if not result or not result[0]:
            print(f"Nenhum resultado encontrado para {symbol}.")
            return None

        result_data = result[0]
        timestamps = result_data.get("timestamp", [])
        indicators = result_data.get("indicators", {})
        quotes = indicators.get("quote", [{}])[0]
        adjclose = indicators.get("adjclose", [{}])[0].get("adjclose", []) if indicators.get("adjclose") else []

        if not timestamps or not quotes.get("close"):
            print(f"Dados incompletos recebidos para {symbol}.")
            return None

        # Cria o DataFrame
        df = pd.DataFrame({
            "Open": quotes.get("open", []), 
            "High": quotes.get("high", []), 
            "Low": quotes.get("low", []), 
            "Close": quotes.get("close", []), 
            "Volume": quotes.get("volume", [])
        }, index=pd.to_datetime(timestamps, unit="s"))

        # Adiciona Adj Close se disponível
        if len(adjclose) == len(df):
             df["Adj Close"] = adjclose
        else:
            print(f"Adj Close data length mismatch for {symbol}, skipping.")

        # Remove linhas onde todos os valores são NaN (pode acontecer em feriados, etc.)        df.dropna(how="all", inplace=True)
        # Remove linhas onde o Volume é NaN ou zero (indicativo de dados faltantes)
        df = df[df["Volume"].notna() & (df["Volume"] > 0)]

        return df

    except Exception as e:
        print(f"Erro inesperado ao buscar ou processar dados para {symbol}: {e}")
        return None

# Coleta e salva os dados para cada símbolo
for name, symbol_code in symbols.items():
    region = "BR" if name == "IBOV" else "US"
    df_market = fetch_yahoo_finance_data(symbol_code, region, data_range, interval)

    if df_market is not None and not df_market.empty:
        file_path = os.path.join(data_dir, f"{name.lower()}_data.csv")
        df_market.to_csv(file_path)
        print(f"Dados de {name} ({symbol_code}) salvos em {file_path}")
    else:
        print(f"Não foi possível obter ou salvar os dados para {name} ({symbol_code}).")
    # Pequena pausa para evitar limites de taxa da API, se houver
    time.sleep(2)

print("\nColeta de dados de mercado (Yahoo Finance) concluída.")

