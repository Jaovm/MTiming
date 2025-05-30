import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime, timedelta
import numpy as np
import requests
import time
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel de Market Timing Brasil",
    page_icon="📊",
    layout="wide"
)

# --- Configurações de APIs ---
API_TOKEN_BRAPI = "5gVedSQ928pxhFuTvBFPfr"  # Token para API brapi.dev
BCB_API_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"

# --- Lista de Tickers (consistente com a coleta) ---
TICKERS = [
    "AGRO3", "BBAS3", "BBSE3", "BPAC11", "EGIE3",
    "ITUB3", "PRIO3", "PSSA3", "SAPR3", "SBSP3",
    "VIVT3", "WEGE3", "TOTS3", "B3SA3", "TAEE3",
    "CMIG3"
]

# Códigos de séries do Banco Central do Brasil
BCB_SERIES = {
    "ipca": 433,         # IPCA - variação % mensal
    "igpm": 189,         # IGP-M - variação % mensal
    "selic_meta": 432,   # Meta Selic definida pelo COPOM - % a.a.
    "selic_efetiva": 11, # Taxa Selic efetiva - % a.a.
    "desemprego": 24369, # Taxa de desocupação - PNAD Contínua - %
    "ibc_br": 24364,     # IBC-Br - Índice (2002=100)
}

# Definição de setores para os tickers
TICKER_SECTORS = {
    "AGRO3": "Agronegócio",
    "BBAS3": "Financeiro",
    "BBSE3": "Seguros",
    "BPAC11": "Financeiro",
    "EGIE3": "Energia",
    "ITUB3": "Financeiro",
    "PRIO3": "Petróleo e Gás",
    "PSSA3": "Seguros",
    "SAPR3": "Saneamento",
    "SBSP3": "Saneamento",
    "VIVT3": "Telecomunicações",
    "WEGE3": "Bens de Capital",
    "TOTS3": "Tecnologia",
    "B3SA3": "Financeiro",
    "TAEE3": "Energia",
    "CMIG3": "Energia"
}

# --- Funções de Coleta de Dados Online ---
@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_bcb_data(series_id, series_name, start_date=None, end_date=None):
    """Busca dados de séries temporais do Banco Central do Brasil."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%d/%m/%Y')  # 10 anos de dados
    if end_date is None:
        end_date = datetime.now().strftime('%d/%m/%Y')
    
    url = f"{BCB_API_BASE_URL}.{series_id}/dados"
    params = {
        'formato': 'json',
        'dataInicial': start_date,
        'dataFinal': end_date
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Converter para DataFrame
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df = df.set_index('data')
        df.columns = [series_name]  # Renomeia a coluna para o nome da série
        
        return df
    except Exception as e:
        st.warning(f"Erro ao buscar dados do BCB para série {series_name} (ID: {series_id}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_yahoo_finance_data(ticker, period="10y", interval="1mo"):
    """Busca dados históricos de preços do Yahoo Finance."""
    # Adiciona sufixo .SA para ações brasileiras se não for um índice
    yahoo_ticker = ticker if ticker.startswith('^') else f"{ticker}.SA"
    
    try:
        # Usar yfinance para obter dados históricos
        data = yf.download(yahoo_ticker, period=period, interval=interval)
        return data
    except Exception as e:
        st.warning(f"Erro ao buscar dados do Yahoo Finance para {ticker}: {e}")
        
        # Fallback para API alternativa se yfinance falhar
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
            params = {
                'range': period,
                'interval': interval,
                'includePrePost': 'false',
                'events': 'div,split'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extrair dados do resultado
            chart_data = data['chart']['result'][0]
            timestamps = chart_data['timestamp']
            quote = chart_data['indicators']['quote'][0]
            
            # Criar DataFrame
            df = pd.DataFrame({
                'Open': quote['open'],
                'High': quote['high'],
                'Low': quote['low'],
                'Close': quote['close'],
                'Volume': quote['volume']
            }, index=pd.to_datetime([datetime.fromtimestamp(x) for x in timestamps]))
            
            # Adicionar Adjusted Close se disponível
            if 'adjclose' in chart_data['indicators']:
                df['Adj Close'] = chart_data['indicators']['adjclose'][0]['adjclose']
            
            return df
        except Exception as e:
            st.warning(f"Erro ao buscar dados do Yahoo Finance (fallback) para {ticker}: {e}")
            return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_brapi_fundamental_data(ticker):
    """Busca dados fundamentalistas de um ticker na API brapi.dev."""
    endpoint = f"/quote/{ticker}"
    base_url = "https://brapi.dev/api"
    headers = {"Authorization": f"Bearer {API_TOKEN_BRAPI}"}
    params = {"fundamental": "true"}
    
    url = base_url + endpoint
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if not data or not data.get("results") or data.get("error"):
            error_msg = data.get("error", "Resposta vazia ou sem resultados.")
            st.warning(f"Erro ao buscar dados para {ticker}: {error_msg}")
            return None
        
        # Extrair os dados fundamentais
        results = data["results"][0]
        return results
    except Exception as e:
        st.warning(f"Erro ao buscar dados fundamentalistas para {ticker}: {e}")
        return None

def filter_data_by_period(df, period_option):
    """Filtra o DataFrame com base no período selecionado."""
    if df.empty:
        return df
    end_date = df.index.max()
    start_date = df.index.min()
    if period_option == "1A":
        start_date = end_date - pd.DateOffset(years=1)
    elif period_option == "3A":
        start_date = end_date - pd.DateOffset(years=3)
    elif period_option == "5A":
        start_date = end_date - pd.DateOffset(years=5)
    elif period_option == "10A":
        start_date = end_date - pd.DateOffset(years=10)
    return df[df.index >= start_date]

# --- Função para Calcular Sinais de Market Timing ---
def calculate_market_timing_signals(df_selic, df_ipca, df_desemprego, df_ibov):
    """Calcula sinais simples de market timing com base nas tendências recentes."""
    signals = {}
    alerts = []
    missing_data = []
    
    # Log de dados disponíveis
    st.session_state['debug_info'] = {
        'selic_data': not df_selic.empty,
        'ipca_data': not df_ipca.empty,
        'desemprego_data': not df_desemprego.empty,
        'ibov_data': not df_ibov.empty,
        'selic_len': len(df_selic) if not df_selic.empty else 0,
        'ipca_len': len(df_ipca) if not df_ipca.empty else 0,
        'desemprego_len': len(df_desemprego) if not df_desemprego.empty else 0,
        'ibov_len': len(df_ibov) if not df_ibov.empty else 0
    }

    # 1. Tendência da Selic (últimos 3 meses)
    if not df_selic.empty and len(df_selic) >= 3:
        selic_trend = df_selic.iloc[-1].iloc[0] - df_selic.iloc[-4].iloc[0] if len(df_selic) >= 4 else 0
        if selic_trend > 0.5: signals['selic'] = 'Alta Forte'
        elif selic_trend > 0.1: signals['selic'] = 'Alta'
        elif selic_trend < -0.5: signals['selic'] = 'Queda Forte'
        elif selic_trend < -0.1: signals['selic'] = 'Queda'
        else: signals['selic'] = 'Estável'
        if signals['selic'] == 'Alta Forte': alerts.append("⚠️ Alerta: Juros subindo agressivamente!")
    else: 
        signals['selic'] = 'N/D'
        missing_data.append('Selic')

    # 2. Tendência da Inflação (IPCA - média móvel 3m vs 6m)
    if not df_ipca.empty and len(df_ipca) >= 6:
        ipca_3m = df_ipca.iloc[-3:].mean().iloc[0]
        ipca_6m = df_ipca.iloc[-6:].mean().iloc[0]
        if ipca_3m > ipca_6m + 0.1: signals['ipca'] = 'Acelerando'
        elif ipca_3m < ipca_6m - 0.1: signals['ipca'] = 'Desacelerando'
        else: signals['ipca'] = 'Estável'
        if signals['ipca'] == 'Acelerando': alerts.append("⚠️ Alerta: Inflação acelerando!")
    else: 
        signals['ipca'] = 'N/D'
        missing_data.append('IPCA')

    # 3. Tendência do Desemprego (últimos 2 trimestres)
    if not df_desemprego.empty and len(df_desemprego) >= 2:
        desemprego_trend = df_desemprego.iloc[-1].iloc[0] - df_desemprego.iloc[-2].iloc[0]
        if desemprego_trend > 0.2: signals['desemprego'] = 'Subindo'
        elif desemprego_trend < -0.2: signals['desemprego'] = 'Caindo'
        else: signals['desemprego'] = 'Estável'
    else: 
        signals['desemprego'] = 'N/D'
        missing_data.append('Desemprego')

    # 4. Tendência do Ibovespa (média móvel 50d vs 200d)
    if not df_ibov.empty and len(df_ibov) >= 200:
        ibov_ma50 = df_ibov['Close'].rolling(window=50).mean().iloc[-1]
        ibov_ma200 = df_ibov['Close'].rolling(window=200).mean().iloc[-1]
        if ibov_ma50 > ibov_ma200 * 1.02: signals['ibov'] = 'Alta (MM50 > MM200)'
        elif ibov_ma50 < ibov_ma200 * 0.98: signals['ibov'] = 'Queda (MM50 < MM200)'
        else: signals['ibov'] = 'Lateral'
    else: 
        signals['ibov'] = 'N/D'
        missing_data.append('Ibovespa')

    # 5. Determinar Fase do Ciclo (Lógica Melhorada e Mais Flexível)
    fase = "Indefinida"
    confianca = 0  # Nível de confiança na determinação da fase
    
    # Verificar se temos dados suficientes para determinar a fase
    dados_suficientes = all(signals.get(key) != 'N/D' for key in ['selic', 'ipca', 'desemprego', 'ibov'])
    
    # Sistema de pontuação para cada fase
    pontos_fase = {
        "Expansão": 0,
        "Pico": 0,
        "Contração": 0,
        "Recuperação": 0
    }
    
    # Avaliar cada indicador e atribuir pontos para as fases
    
    # Selic
    if signals.get('selic') in ['Queda', 'Queda Forte']:
        pontos_fase["Expansão"] += 1
        pontos_fase["Recuperação"] += 1
    elif signals.get('selic') in ['Alta', 'Alta Forte']:
        pontos_fase["Pico"] += 1
        pontos_fase["Contração"] += 1
    elif signals.get('selic') == 'Estável':
        pontos_fase["Expansão"] += 0.5
        pontos_fase["Pico"] += 0.5
        pontos_fase["Contração"] += 0.5
        pontos_fase["Recuperação"] += 0.5
    
    # IPCA
    if signals.get('ipca') == 'Acelerando':
        pontos_fase["Pico"] += 1
    elif signals.get('ipca') == 'Desacelerando':
        pontos_fase["Contração"] += 1
        pontos_fase["Recuperação"] += 1
    elif signals.get('ipca') == 'Estável':
        pontos_fase["Expansão"] += 1
        pontos_fase["Recuperação"] += 0.5
    
    # Desemprego
    if signals.get('desemprego') == 'Caindo':
        pontos_fase["Expansão"] += 1
    elif signals.get('desemprego') == 'Subindo':
        pontos_fase["Contração"] += 1
        pontos_fase["Recuperação"] += 0.5
    elif signals.get('desemprego') == 'Estável':
        pontos_fase["Pico"] += 1
        pontos_fase["Expansão"] += 0.5
    
    # Ibovespa
    if signals.get('ibov') == 'Alta (MM50 > MM200)':
        pontos_fase["Expansão"] += 1
        pontos_fase["Pico"] += 0.5
    elif signals.get('ibov') == 'Queda (MM50 < MM200)':
        pontos_fase["Contração"] += 1
    elif signals.get('ibov') == 'Lateral':
        pontos_fase["Pico"] += 0.5
        pontos_fase["Recuperação"] += 1
    
    # Determinar a fase com maior pontuação
    if not dados_suficientes:
        # Se faltam dados, ainda podemos tentar determinar a fase com os dados disponíveis
        # mas com menor confiança
        max_pontos = max(pontos_fase.values())
        if max_pontos > 0:
            # Encontrar todas as fases com pontuação máxima
            fases_max = [f for f, p in pontos_fase.items() if p == max_pontos]
            if len(fases_max) == 1:
                fase = f"Provável {fases_max[0]}"
                confianca = max_pontos / 4  # Normalizar para 0-1
            else:
                # Se houver empate, escolher com base em prioridades
                prioridades = ["Contração", "Recuperação", "Expansão", "Pico"]
                for p in prioridades:
                    if p in fases_max:
                        fase = f"Possível {p}"
                        confianca = max_pontos / 4  # Normalizar para 0-1
                        break
        
        # Adicionar informação sobre dados faltantes
        if missing_data:
            fase += f" (Dados faltantes: {', '.join(missing_data)})"
    else:
        # Com todos os dados disponíveis, temos maior confiança
        max_pontos = max(pontos_fase.values())
        fase_max = max(pontos_fase, key=pontos_fase.get)
        confianca = max_pontos / 4  # Normalizar para 0-1
        
        # Classificar com base na confiança
        if confianca >= 0.75:
            fase = fase_max
        elif confianca >= 0.5:
            fase = f"Provável {fase_max}"
        else:
            fase = f"Possível {fase_max}"
    
    # Armazenar informações detalhadas para debug
    st.session_state['fase_debug'] = {
        'pontos_fase': pontos_fase,
        'confianca': confianca,
        'dados_suficientes': dados_suficientes,
        'missing_data': missing_data
    }
    
    signals['fase_ciclo'] = fase
    signals['confianca_fase'] = confianca

    # Adicionar VIX (se disponível)
    df_vix = fetch_yahoo_finance_data("^VIX", period="1mo")
    if not df_vix.empty:
        last_vix = df_vix['Close'].iloc[-1]
        signals['vix'] = f"{last_vix:.2f}"
        if last_vix > 30: # Exemplo de limite para alerta
            alerts.append(f"⚠️ Alerta: VIX alto ({last_vix:.2f})!")
    else:
        signals['vix'] = 'N/D'

    return signals, alerts

# --- Função para identificar ciclos econômicos semelhantes ---
def identify_similar_cycles(current_signals, historical_data):
    """Identifica períodos históricos com características semelhantes ao ciclo atual."""
    if not historical_data or not current_signals:
        return []
    
    # Extrair características atuais
    current_features = {
        'selic': 0,
        'ipca': 0,
        'desemprego': 0,
        'ibov': 0
    }
    
    # Codificar características atuais
    if current_signals.get('selic') in ['Alta Forte', 'Alta']:
        current_features['selic'] = 1
    elif current_signals.get('selic') in ['Queda Forte', 'Queda']:
        current_features['selic'] = -1
    
    if current_signals.get('ipca') == 'Acelerando':
        current_features['ipca'] = 1
    elif current_signals.get('ipca') == 'Desacelerando':
        current_features['ipca'] = -1
    
    if current_signals.get('desemprego') == 'Subindo':
        current_features['desemprego'] = 1
    elif current_signals.get('desemprego') == 'Caindo':
        current_features['desemprego'] = -1
    
    if current_signals.get('ibov') == 'Alta (MM50 > MM200)':
        current_features['ibov'] = 1
    elif current_signals.get('ibov') == 'Queda (MM50 < MM200)':
        current_features['ibov'] = -1
    
    # Preparar para comparação com períodos históricos
    similar_periods = []
    
    # Analisar dados históricos por janelas de 6 meses
    for year in range(2010, datetime.now().year):
        for month in range(1, 13, 6):  # Janelas de 6 meses
            try:
                per
