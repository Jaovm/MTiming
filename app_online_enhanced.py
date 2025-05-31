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
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Verificar se a resposta não está vazia
        if not response.text.strip():
            st.warning(f"Resposta vazia do BCB para série {series_name} (ID: {series_id})")
            return pd.DataFrame()
            
        # Tentar converter para JSON com tratamento de erro
        try:
            data = response.json()
        except json.JSONDecodeError as json_err:
            st.warning(f"Erro ao decodificar JSON do BCB para série {series_name} (ID: {series_id}): {json_err}")
            return pd.DataFrame()
            
        # Verificar se data é uma lista válida e não vazia
        if not isinstance(data, list) or len(data) == 0:
            st.warning(f"Dados inválidos do BCB para série {series_name} (ID: {series_id}): formato inesperado")
            return pd.DataFrame()
        
        # Converter para DataFrame
        df = pd.DataFrame(data)
        
        # Verificar se as colunas esperadas existem
        if 'data' not in df.columns or 'valor' not in df.columns:
            st.warning(f"Colunas esperadas não encontradas nos dados do BCB para série {series_name} (ID: {series_id})")
            return pd.DataFrame()
            
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # Remover linhas com datas inválidas
        df = df.dropna(subset=['data'])
        
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
    if not df_vix.empty and len(df_vix) > 0:
        try:
            last_vix_val = df_vix['Close'].iloc[-1]
            # Garantir que last_vix_val seja um valor escalar
            if isinstance(last_vix_val, pd.Series):
                last_vix_val = last_vix_val.iloc[0] if not last_vix_val.empty else None
            
            if last_vix_val is not None and pd.notna(last_vix_val):
                last_vix_float = float(last_vix_val)
                signals['vix'] = f"{last_vix_float:.2f}"
                if last_vix_float > 30: # Exemplo de limite para alerta
                    alerts.append(f"⚠️ Alerta: VIX alto ({last_vix_float:.2f})!")
            else:
                signals['vix'] = 'N/D'
        except (TypeError, ValueError, IndexError) as e:
            st.warning(f"Erro ao processar valor do VIX: {e}")
            signals['vix'] = 'N/D'
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
                period_start = datetime(year, month, 1)
                period_end = period_start + timedelta(days=180)
                
                # Pular períodos muito recentes (menos de 1 ano atrás)
                if period_end > datetime.now() - timedelta(days=365):
                    continue
                
                # Extrair características do período
                period_features = {
                    'selic': 0,
                    'ipca': 0,
                    'desemprego': 0,
                    'ibov': 0
                }
                
                # Analisar tendência da Selic no período
                selic_data = historical_data.get('selic_meta')
                if not selic_data.empty:
                    period_selic = selic_data[(selic_data.index >= period_start) & (selic_data.index <= period_end)]
                    if not period_selic.empty and len(period_selic) > 1:
                        selic_trend = period_selic.iloc[-1].iloc[0] - period_selic.iloc[0].iloc[0]
                        if selic_trend > 0.5:
                            period_features['selic'] = 1
                        elif selic_trend < -0.5:
                            period_features['selic'] = -1
                
                # Analisar tendência do IPCA no período
                ipca_data = historical_data.get('ipca')
                if not ipca_data.empty:
                    period_ipca = ipca_data[(ipca_data.index >= period_start) & (ipca_data.index <= period_end)]
                    if not period_ipca.empty and len(period_ipca) > 2:
                        ipca_first_half = period_ipca.iloc[:len(period_ipca)//2].mean().iloc[0]
                        ipca_second_half = period_ipca.iloc[len(period_ipca)//2:].mean().iloc[0]
                        if ipca_second_half > ipca_first_half + 0.1:
                            period_features['ipca'] = 1
                        elif ipca_second_half < ipca_first_half - 0.1:
                            period_features['ipca'] = -1
                
                # Analisar tendência do desemprego no período
                desemprego_data = historical_data.get('desemprego')
                if not desemprego_data.empty:
                    period_desemprego = desemprego_data[(desemprego_data.index >= period_start) & (desemprego_data.index <= period_end)]
                    if not period_desemprego.empty and len(period_desemprego) > 1:
                        desemprego_trend = period_desemprego.iloc[-1].iloc[0] - period_desemprego.iloc[0].iloc[0]
                        if desemprego_trend > 0.2:
                            period_features['desemprego'] = 1
                        elif desemprego_trend < -0.2:
                            period_features['desemprego'] = -1
                
                # Analisar tendência do Ibovespa no período
                ibov_data = historical_data.get('ibovespa')
                if not ibov_data.empty:
                    period_ibov = ibov_data[(ibov_data.index >= period_start) & (ibov_data.index <= period_end)]
                    if not period_ibov.empty and len(period_ibov) > 50:
                        period_ibov_ma50 = period_ibov['Close'].rolling(window=min(50, len(period_ibov))).mean().iloc[-1]
                        period_ibov_ma200 = period_ibov['Close'].rolling(window=min(200, len(period_ibov))).mean().iloc[-1]
                        if period_ibov_ma50 > period_ibov_ma200 * 1.02:
                            period_features['ibov'] = 1
                        elif period_ibov_ma50 < period_ibov_ma200 * 0.98:
                            period_features['ibov'] = -1
                
                # Calcular similaridade
                similarity_score = sum(1 for k in current_features if current_features[k] == period_features[k] and current_features[k] != 0)
                total_features = sum(1 for k in current_features if current_features[k] != 0)
                
                if total_features > 0:
                    similarity = similarity_score / total_features
                    
                    # Adicionar período se similaridade for alta
                    if similarity >= 0.5:
                        similar_periods.append({
                            'period': f"{period_start.strftime('%b/%Y')} - {period_end.strftime('%b/%Y')}",
                            'similarity': similarity,
                            'features': period_features
                        })
            except Exception as e:
                st.warning(f"Erro ao analisar período {year}/{month}: {e}")
    
    # Ordenar por similaridade
    similar_periods.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_periods[:3]  # Retornar os 3 períodos mais similares

# --- Função para identificar tickers com melhor desempenho em ciclos semelhantes ---
def identify_best_performers(similar_periods, tickers=TICKERS):
    """Identifica tickers com melhor desempenho em ciclos econômicos semelhantes."""
    if not similar_periods:
        return []
    
    best_performers = []
    
    for period_info in similar_periods:
        period_str = period_info['period']
        period_start_str, period_end_str = period_str.split(' - ')
        
        # Converter para datetime
        try:
            period_start = datetime.strptime(period_start_str, '%b/%Y')
            period_end = datetime.strptime(period_end_str, '%b/%Y')
            
            # Estender o período para capturar o desempenho após o ciclo
            performance_end = period_end + timedelta(days=180)  # 6 meses após o fim do ciclo
            
            # Calcular desempenho para cada ticker
            ticker_performances = []
            
            for ticker in tickers:
                try:
                    # Buscar dados históricos
                    ticker_data = fetch_yahoo_finance_data(ticker, period="max", interval="1mo")
                    
                    if not ticker_data.empty:
                        # Filtrar para o período relevante
                        period_data = ticker_data[(ticker_data.index >= period_start) & (ticker_data.index <= performance_end)]
                        
                        if not period_data.empty and len(period_data) > 1:
                            # Calcular retorno
                            start_price = period_data['Close'].iloc[0]
                            end_price = period_data['Close'].iloc[-1]
                            
                            if start_price > 0:
                                performance = ((end_price / start_price) - 1) * 100
                                ticker_performances.append({
                                    'ticker': ticker,
                                    'performance': performance,
                                    'sector': TICKER_SECTORS.get(ticker, "Outros")
                                })
                except Exception as e:
                    st.warning(f"Erro ao calcular desempenho para {ticker} no período {period_str}: {e}")
            
            # Ordenar por desempenho
            ticker_performances.sort(key=lambda x: x['performance'], reverse=True)
            
            # Adicionar à lista de melhores performers
            best_performers.append({
                'period': period_str,
                'similarity': period_info['similarity'],
                'performers': ticker_performances[:5]  # Top 5 performers
            })
        except Exception as e:
            st.warning(f"Erro ao processar período {period_str}: {e}")
    
    return best_performers

# --- Função para calcular preço justo por diferentes modelos ---
def calculate_fair_price(ticker_data, historical_data=None):
    """Calcula o preço justo da ação usando diferentes modelos de valuation."""
    if not ticker_data:
        return None
    
    valuation_models = {}
    
    # Extrair dados básicos
    current_price = ticker_data.get('regularMarketPrice')
    if not current_price:
        return None
    
    # 1. Modelo P/L (Price-to-Earnings)
    try:
        earnings_per_share = ticker_data.get('earningsPerShare')
        if earnings_per_share and earnings_per_share > 0:
            # Usar P/L médio do setor ou um valor padrão
            sector_pe = 15  # Valor médio para o mercado brasileiro
            pe_fair_price = earnings_per_share * sector_pe
            valuation_models['P/L'] = pe_fair_price
    except Exception as e:
        st.warning(f"Erro no cálculo do modelo P/L: {e}")
    
    # 2. Modelo P/VP (Price-to-Book)
    try:
        # Tentar calcular VPA a partir de EPS (estimativa)
        if 'earningsPerShare' in ticker_data and ticker_data['earningsPerShare']:
            eps = ticker_data['earningsPerShare']
            estimated_vpa = eps * 10  # Estimativa grosseira
            
            # Usar P/VP médio do setor ou um valor padrão
            sector_pb = 2.0  # Valor médio para o mercado brasileiro
            pb_fair_price = estimated_vpa * sector_pb
            valuation_models['P/VP'] = pb_fair_price
    except Exception as e:
        st.warning(f"Erro no cálculo do modelo P/VP: {e}")
    
    # 3. Modelo de Crescimento de Dividendos (Gordon)
    try:
        if historical_data and ticker in historical_data:
            ticker_hist = historical_data[ticker]
            if not ticker_hist.empty and 'Close' in ticker_hist.columns:
                # Estimar taxa de crescimento com base no histórico de preços
                growth_rate = 0.05  # Taxa de crescimento padrão (5%)
                
                # Estimar dividend yield
                div_yield = 0.03  # Dividend yield padrão (3%)
                
                # Taxa de desconto (CAPM simplificado)
                risk_free_rate = 0.10  # Taxa livre de risco (Selic)
                market_premium = 0.05  # Prêmio de risco de mercado
                beta = 1.0  # Beta padrão
                
                discount_rate = risk_free_rate + beta * market_premium
                
                # Modelo de Gordon
                if discount_rate > growth_rate:
                    gordon_fair_price = current_price * div_yield / (discount_rate - growth_rate)
                    valuation_models['Gordon'] = gordon_fair_price
    except Exception as e:
        st.warning(f"Erro no cálculo do modelo de Gordon: {e}")
    
    # 4. Modelo de Múltiplos de Mercado
    try:
        if 'marketCap' in ticker_data and ticker_data['marketCap']:
            market_cap = ticker_data['marketCap']
            
            # Estimar receita
            if 'earningsPerShare' in ticker_data and ticker_data['earningsPerShare']:
                eps = ticker_data['earningsPerShare']
                estimated_shares = market_cap / current_price
                estimated_earnings = eps * estimated_shares
                
                # Usar múltiplo P/S (Price-to-Sales) médio do setor
                sector_ps = 2.0  # Valor médio para o mercado brasileiro
                ps_fair_price = (estimated_earnings * 10) * sector_ps / estimated_shares  # Receita estimada como 10x lucro
                valuation_models['Múltiplos'] = ps_fair_price
    except Exception as e:
        st.warning(f"Erro no cálculo do modelo de Múltiplos: {e}")
    
    # Calcular média e mediana se houver pelo menos um modelo
    if valuation_models:
        prices = list(valuation_models.values())
        mean_price = sum(prices) / len(prices)
        median_price = sorted(prices)[len(prices) // 2] if len(prices) % 2 == 1 else sum(sorted(prices)[len(prices) // 2 - 1:len(prices) // 2 + 1]) / 2
        
        return {
            'models': valuation_models,
            'mean': mean_price,
            'median': median_price,
            'current_price': current_price,
            'upside_mean': ((mean_price / current_price) - 1) * 100,
            'upside_median': ((median_price / current_price) - 1) * 100
        }
    
    return None

# --- Função para Gerar Recomendações ---
def generate_recommendation(ticker_data, cycle_phase, valuation_result=None):
    """Gera recomendação com base nos múltiplos atuais, fase do ciclo e valuation."""
    if not ticker_data:
        return "N/D", "Dados fundamentalistas indisponíveis."

    # Extrair múltiplos (tratar possíveis Nones ou erros de conversão)
    pl = None
    pvp = None
    try:
        # Tenta obter e converter P/L. Verifica múltiplos campos possíveis.
        # Primeiro tenta priceEarnings (formato da API brapi)
        raw_pl = ticker_data.get("priceEarnings")
        # Se não encontrar, tenta trailingPE (formato alternativo)
        if raw_pl is None:
            raw_pl = ticker_data.get("trailingPE")
        if raw_pl is not None:
            pl = float(raw_pl)
    except (ValueError, TypeError):
        pl = None # Garante que pl é None se a conversão falhar
    
    try:
        # Para P/VP, não há campo direto nos dados atuais
        # Podemos calcular se tivermos preço e valor patrimonial por ação
        raw_pvp = ticker_data.get("priceToBook")
        
        # Se não encontrar diretamente, tenta calcular a partir do preço e VPA
        if raw_pvp is None and "regularMarketPrice" in ticker_data and "earningsPerShare" in ticker_data:
            price = ticker_data.get("regularMarketPrice")
            eps = ticker_data.get("earningsPerShare")
            # Estimativa simples: assumindo que VPA é aproximadamente 10x EPS para empresas brasileiras
            # Esta é uma aproximação grosseira e deve ser substituída por dados reais quando disponíveis
            if price is not None and eps is not None and eps != 0:
                estimated_vpa = eps * 10  # Estimativa grosseira
                raw_pvp = price / estimated_vpa
        
        if raw_pvp is not None:
            pvp = float(raw_pvp)
    except (ValueError, TypeError, ZeroDivisionError):
        pvp = None # Garante que pvp é None se a conversão falhar

    # Formatar múltiplos para exibição (N/D se None)
    pl_display = f"{pl:.1f}" if pl is not None else "N/D"
    pvp_display = f"{pvp:.1f}" if pvp is not None else "N/D"

    # Definir Níveis de Valuation (somente se múltiplos disponíveis)
    val_pl = "N/D"
    val_pvp = "N/D"
    if pl is not None:
        val_pl = "Alto" if pl > 20 else ("Médio" if pl > 10 else "Baixo")
    if pvp is not None:
        val_pvp = "Alto" if pvp > 2 else ("Médio" if pvp > 1 else "Baixo")

    # Lógica de Recomendação
    recomendacao = "Neutro"
    justificativa = f"P/L: {pl_display} ({val_pl}), P/VP: {pvp_display} ({val_pvp}). Fase do ciclo: {cycle_phase}."

    # Se algum múltiplo essencial for N/D, a recomendação é N/D
    if pl is None and pvp is None:
        recomendacao = "N/D"
        justificativa = f"Dados de múltiplos (P/L={pl_display}, P/VP={pvp_display}) indisponíveis ou inválidos. Fase: {cycle_phase}."
        return recomendacao, justificativa # Retorna imediatamente
    
    # Se temos pelo menos um múltiplo, podemos gerar uma recomendação
    has_pl = pl is not None
    has_pvp = pvp is not None
    
    # Extrair a fase base do ciclo (remover "Provável", "Possível", etc.)
    base_cycle_phase = cycle_phase
    for prefix in ["Provável ", "Possível ", "Indefinida"]:
        if cycle_phase.startswith(prefix):
            base_cycle_phase = cycle_phase.replace(prefix, "")
            if "(" in base_cycle_phase:
                base_cycle_phase = base_cycle_phase.split("(")[0].strip()
            break

    # Lógica baseada na fase e valuation
    if "Expansão" in base_cycle_phase:
        if (has_pl and val_pl == "Baixo") or (has_pvp and val_pvp == "Baixo"):
            recomendacao = "Compra Forte"
        elif (has_pl and val_pl != "Alto") or (has_pvp and val_pvp != "Alto"):
            recomendacao = "Compra"
        else:
            recomendacao = "Neutro"
    elif "Pico" in base_cycle_phase:
        if (has_pl and val_pl == "Alto") or (has_pvp and val_pvp == "Alto"):
            recomendacao = "Venda"
        elif (has_pl and val_pl == "Médio") or (has_pvp and val_pvp == "Médio"):
            recomendacao = "Neutro/Venda"
        else:
            recomendacao = "Neutro"
    elif "Contração" in base_cycle_phase:
        if (has_pl and val_pl == "Alto") or (has_pvp and val_pvp == "Alto"):
            recomendacao = "Venda Forte"
        elif (has_pl and val_pl == "Médio") or (has_pvp and val_pvp == "Médio"):
            recomendacao = "Venda"
        else:
            recomendacao = "Neutro/Avaliar"
    elif "Recuperação" in base_cycle_phase:
        if (has_pl and val_pl == "Baixo") or (has_pvp and val_pvp == "Baixo"):
            recomendacao = "Compra"
        elif (has_pl and val_pl != "Alto") or (has_pvp and val_pvp != "Alto"):
            recomendacao = "Compra/Neutro"
        else:
            recomendacao = "Neutro"
    else:
        # Se fase indefinida ou não reconhecida, recomendação é Neutro
        recomendacao = "Neutro"
        # Ajusta a justificativa para ser mais clara sobre a fase
        justificativa = f"Fase do ciclo {cycle_phase}. "
        if has_pl:
            justificativa += f"P/L: {pl_display} ({val_pl}), "
        if has_pvp:
            justificativa += f"P/VP: {pvp_display} ({val_pvp})."
        else:
            justificativa = justificativa.rstrip(", ") + "."

    # Incorporar informações de valuation se disponíveis
    if valuation_result:
        upside_mean = valuation_result.get('upside_mean')
        if upside_mean is not None:
            # Ajustar recomendação com base no upside potencial
            if upside_mean > 30:
                if recomendacao in ["Neutro", "Neutro/Avaliar", "Compra/Neutro"]:
                    recomendacao = "Compra"
                elif recomendacao == "Compra":
                    recomendacao = "Compra Forte"
            elif upside_mean < -20:
                if recomendacao in ["Neutro", "Neutro/Avaliar", "Neutro/Venda"]:
                    recomendacao = "Venda"
                elif recomendacao == "Venda":
                    recomendacao = "Venda Forte"
            
            # Adicionar informação de valuation à justificativa
            justificativa += f" Preço justo médio: R${valuation_result['mean']:.2f} (Upside: {upside_mean:.1f}%)."

    # Atualiza a justificativa final
    if recomendacao != "N/D":
        if "Preço justo" not in justificativa:
            justificativa = f"P/L: {pl_display} ({val_pl if has_pl else 'N/D'}), P/VP: {pvp_display} ({val_pvp if has_pvp else 'N/D'}). Fase do ciclo: {cycle_phase}. Recomendação: {recomendacao}."

    return recomendacao, justificativa

# --- Carregamento de Dados Online ---
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_all_macro_data():
    """Carrega todos os dados macroeconômicos necessários."""
    with st.spinner('Carregando dados macroeconômicos...'):
        # Dicionário para armazenar todos os DataFrames
        macro_data = {}
        
        # Coletar dados do BCB
        for name, series_id in BCB_SERIES.items():
            df = fetch_bcb_data(series_id, name)
            if not df.empty:
                macro_data[name] = df
        
        # Coletar Ibovespa
        ibov_df = fetch_yahoo_finance_data("^BVSP", period="10y")
        if not ibov_df.empty:
            macro_data["ibovespa"] = ibov_df
        
        # Coletar VIX
        vix_df = fetch_yahoo_finance_data("^VIX", period="10y")
        if not vix_df.empty:
            macro_data["vix"] = vix_df
        
        return macro_data

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_all_fundamental_data():
    """Carrega dados fundamentalistas para todos os tickers."""
    with st.spinner('Carregando dados fundamentalistas...'):
        fundamental_data = {}
        
        for ticker in TICKERS:
            data = fetch_brapi_fundamental_data(ticker)
            if data:
                fundamental_data[ticker] = data
            time.sleep(0.5)  # Pausa para evitar limites de taxa
        
        return fundamental_data

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_historical_ticker_data():
    """Carrega dados históricos para todos os tickers."""
    with st.spinner('Carregando dados históricos de ações...'):
        historical_data = {}
        
        for ticker in TICKERS:
            data = fetch_yahoo_finance_data(ticker, period="10y", interval="1mo")
            if not data.empty:
                historical_data[ticker] = data
            time.sleep(0.5)  # Pausa para evitar limites de taxa
        
        return historical_data

# --- Interface do Usuário ---
def main():
    # --- Barra Lateral (Sidebar) ---
    st.sidebar.title("Painel de Market Timing")
    st.sidebar.markdown("**Foco: Brasil**")
    pais_selecionado = st.sidebar.selectbox("País", ["Brasil"], index=0, disabled=True)
    st.sidebar.markdown("--- ")
    st.sidebar.header("Seleção de Ação (Valuation)")
    acao_selecionada_valuation = st.sidebar.selectbox(
        "Escolha uma Ação",
        options=TICKERS,
        index=0,
        help="Selecione a ação para ver os dados fundamentalistas atuais."
    )
    st.sidebar.markdown("--- ")
    st.sidebar.header("Controles Gerais")
    periodo_historico = st.sidebar.select_slider(
        "Período Histórico (Macro)",
        options=["1A", "3A", "5A", "10A", "Máx"],
        value="5A",
        help="Selecione o período para visualização dos gráficos macroeconômicos."
    )
    
    # Opção para mostrar dados de debug
    mostrar_debug = st.sidebar.checkbox("Mostrar dados de debug", value=False)
    
    if st.sidebar.button("Atualizar Dados"):
        st.cache_data.clear()
        st.experimental_rerun()
    last_update_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    st.sidebar.info(f"Última atualização: {last_update_time}")
    st.sidebar.caption("Desenvolvido por Manus")

    # --- Carregamento de Dados ---
    macro_data = load_all_macro_data()
    fundamental_data = load_all_fundamental_data()
    historical_ticker_data = load_historical_ticker_data()
    
    # Extrair DataFrames específicos
    df_ipca = macro_data.get('ipca', pd.DataFrame())
    df_igpm = macro_data.get('igpm', pd.DataFrame())
    df_selic_meta = macro_data.get('selic_meta', pd.DataFrame())
    df_selic_efetiva = macro_data.get('selic_efetiva', pd.DataFrame())
    df_desemprego = macro_data.get('desemprego', pd.DataFrame())
    df_ibov = macro_data.get('ibovespa', pd.DataFrame())
    df_ibc_br = macro_data.get('ibc_br', pd.DataFrame())
    df_vix = macro_data.get('vix', pd.DataFrame())

    # Calcula sinais de timing
    timing_signals, timing_alerts = calculate_market_timing_signals(df_selic_meta, df_ipca, df_desemprego, df_ibov)
    current_cycle_phase = timing_signals.get('fase_ciclo', 'Indefinida')
    
    # Identificar ciclos econômicos semelhantes
    similar_cycles = identify_similar_cycles(timing_signals, macro_data)
    
    # Identificar tickers com melhor desempenho em ciclos semelhantes
    best_performers = identify_best_performers(similar_cycles, TICKERS)

    # --- Área Principal (Abas) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cenário Macroeconômico",
        "📈 Valuation (Atual)",
        "🚦 Sinais de Market Timing",
        "🧭 Alocação Sugerida",
        "⏱️ Timing Histórico"
    ])

    # --- Aba 1: Cenário Macroeconômico ---
    with tab1:
        st.header("Cenário Macroeconômico - Brasil")
        st.subheader("Indicadores Chave")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if not df_ipca.empty:
                last_ipca = df_ipca.iloc[-1].iloc[0]
                prev_ipca = df_ipca.iloc[-2].iloc[0] if len(df_ipca) > 1 else last_ipca
                delta_ipca = last_ipca - prev_ipca
                st.metric("IPCA (Último Mensal)", f"{last_ipca:.2f}%", f"{delta_ipca:.2f} p.p.", help="Índice Nacional de Preços ao Consumidor Amplo. Variação percentual mensal.")
            else:
                st.metric("IPCA (Último Mensal)", "N/D")
            if not df_selic_meta.empty:
                last_selic = df_selic_meta.iloc[-1].iloc[0]
                st.metric("Selic Meta", f"{last_selic:.2f}%", help="Taxa básica de juros definida pelo COPOM (anualizada).")
            else:
                st.metric("Selic Meta", "N/D")
        with col2:
            if not df_igpm.empty:
                last_igpm = df_igpm.iloc[-1].iloc[0]
                prev_igpm = df_igpm.iloc[-2].iloc[0] if len(df_igpm) > 1 else last_igpm
                delta_igpm = last_igpm - prev_igpm
                st.metric("IGP-M (Último Mensal)", f"{last_igpm:.2f}%", f"{delta_igpm:.2f} p.p.", help="Índice Geral de Preços - Mercado. Variação percentual mensal.")
            else:
                st.metric("IGP-M (Último Mensal)", "N/D")
            if not df_selic_efetiva.empty:
                last_selic_efetiva = df_selic_efetiva.iloc[-1].iloc[0]
                st.metric("Selic Efetiva", f"{last_selic_efetiva:.2f}%", help="Taxa média diária dos financiamentos no SELIC (anualizada).")
            else:
                st.metric("Selic Efetiva", "N/D")
        with col3:
            if not df_desemprego.empty:
                last_desemprego = df_desemprego.iloc[-1].iloc[0]
                st.metric("Desemprego PNAD", f"{last_desemprego:.1f}%", help="Taxa de desocupação (último trimestre disponível - IBGE).")
            else:
                st.metric("Desemprego PNAD", "N/D")
            if not df_ibov.empty:
                last_ibov_val = df_ibov_hist['Close'].iloc[-1] if len(df_ibov_hist) >= 1 else None
                prev_ibov_val = df_ibov_hist['Close'].iloc[-2] if len(df_ibov_hist) >= 2 else None
                delta_ibov = 0.0
                if prev_ibov_val is not None and pd.notna(prev_ibov_val) and prev_ibov_val != 0:
                    if last_ibov_val is not None and pd.notna(last_ibov_val):
                        try:
                            delta_ibov = ((float(last_ibov_val) / float(prev_ibov_val)) - 1) * 100
                        except (TypeError, ValueError):
                            st.warning(f"Error calculating delta_ibov: last={last_ibov_val}, prev={prev_ibov_val}")
                    else:
                        st.warning(f"last_ibov_val is invalid: {last_ibov_val}")
                elif prev_ibov_val == 0:
                    st.warning("prev_ibov_val is zero, cannot calculate delta.")
                # else: prev_ibov_val is None or NaN, delta_ibov remains 0.0
                st.metric("Ibovespa (Fechamento)", f"{last_ibov_val:,.0f}", f"{delta_ibov:.2f}%", delta_color="normal", help="Último valor de fechamento do Índice Bovespa.")
            else:
                st.metric("Ibovespa (Fechamento)", "N/D")
        with col4:
            if not df_ibc_br.empty:
                last_ibc = df_ibc_br.iloc[-1].iloc[0]
                st.metric("IBC-Br (Proxy PIB)", f"{last_ibc:.2f}", help="Índice de Atividade Econômica do Banco Central.")
            else:
                st.metric("IBC-Br (Proxy PIB)", "N/D", help="Dados não disponíveis.")
            if not df_vix.empty:
                last_vix = df_vix['Close'].iloc[-1]
                st.metric("VIX", f"{last_vix:.2f}", help="Índice de Volatilidade CBOE.")
            else:
                st.metric("VIX", "N/D", help="Dados não disponíveis.")
        st.divider()
        st.subheader("Gráficos Históricos (Macro)")
        df_ipca_hist = filter_data_by_period(df_ipca, periodo_historico)
        df_igpm_hist = filter_data_by_period(df_igpm, periodo_historico)
        df_selic_hist = filter_data_by_period(df_selic_meta, periodo_historico)
        df_desemprego_hist = filter_data_by_period(df_desemprego, periodo_historico)
        df_ibov_hist = filter_data_by_period(df_ibov, periodo_historico)
        if not df_ipca_hist.empty or not df_igpm_hist.empty:
            fig_inflacao = go.Figure()
            if not df_ipca_hist.empty:
                fig_inflacao.add_trace(go.Scatter(x=df_ipca_hist.index, y=df_ipca_hist[df_ipca_hist.columns[0]], mode='lines', name='IPCA (% Mensal)'))
            if not df_igpm_hist.empty:
                fig_inflacao.add_trace(go.Scatter(x=df_igpm_hist.index, y=df_igpm_hist[df_igpm_hist.columns[0]], mode='lines', name='IGP-M (% Mensal)'))
            fig_inflacao.update_layout(title='Inflação Mensal (IPCA vs IGP-M)', yaxis_title='%', legend_title="Índice")
            st.plotly_chart(fig_inflacao, use_container_width=True)
        else:
            st.write("Dados de inflação não disponíveis para o período.")
        if not df_selic_hist.empty:
            fig_selic = px.line(df_selic_hist, y=df_selic_hist.columns[0], title='Taxa Selic Meta (% a.a.)')
            fig_selic.update_layout(yaxis_title='% a.a.', legend_title="Taxa")
            st.plotly_chart(fig_selic, use_container_width=True)
        else:
            st.write("Dados da Selic não disponíveis para o período.")
        if not df_ibov_hist.empty:
            fig_ibov = px.line(df_ibov_hist, y='Close', title='Ibovespa (Fechamento)')
            fig_ibov.update_layout(yaxis_title='Pontos', legend_title="Índice")
            st.plotly_chart(fig_ibov, use_container_width=True)
        else:
            st.write("Dados do Ibovespa não disponíveis para o período.")

    # --- Aba 2: Valuation (Atual) ---
    with tab2:
        st.header(f"Valuation Atual - {acao_selecionada_valuation}")
        st.markdown("Esta aba exibe os indicadores fundamentalistas atuais da ação selecionada e o preço justo estimado por diferentes modelos de valuation.")
        
        ticker_data = fundamental_data.get(acao_selecionada_valuation)
        
        if ticker_data:
            # Calcular preço justo
            valuation_result = calculate_fair_price(ticker_data, historical_ticker_data.get(acao_selecionada_valuation))
            
            st.subheader("Indicadores Principais")
            col1, col2 = st.columns(2)
            
            with col1:
                # Preço e Valor de Mercado
                price = ticker_data.get("regularMarketPrice", "N/D")
                st.metric("Preço Atual", f"{price}")
                
                market_cap = ticker_data.get("marketCap", "N/D")
                st.metric("Valor de Mercado", f"{market_cap}")
                
                # P/L
                try:
                    pl = float(ticker_data.get("priceEarnings", ticker_data.get("trailingPE", "N/D")))
                    st.metric("P/L (12m)", f"{pl:.2f}")
                except (ValueError, TypeError):
                    st.metric("P/L (12m)", "N/D")
                
                # P/L Projetado (se disponível)
                st.metric("P/L (Proj.)", "N/D")
            
            with col2:
                # P/VP
                try:
                    pvp = float(ticker_data.get("priceToBook", "N/D"))
                    st.metric("P/VP", f"{pvp:.2f}")
                except (ValueError, TypeError):
                    # Tenta calcular P/VP a partir de outros dados
                    try:
                        if "regularMarketPrice" in ticker_data and "earningsPerShare" in ticker_data:
                            price = ticker_data.get("regularMarketPrice")
                            eps = ticker_data.get("earningsPerShare")
                            if price is not None and eps is not None and eps != 0:
                                estimated_vpa = eps * 10  # Estimativa grosseira
                                pvp = price / estimated_vpa
                                st.metric("P/VP (estimado)", f"{pvp:.2f}")
                            else:
                                st.metric("P/VP", "N/D")
                        else:
                            st.metric("P/VP", "N/D")
                    except:
                        st.metric("P/VP", "N/D")
                
                # VPA
                try:
                    eps = float(ticker_data.get("earningsPerShare", "N/D"))
                    estimated_vpa = eps * 10  # Estimativa grosseira
                    st.metric("VPA (estimado)", f"{estimated_vpa:.2f}")
                except (ValueError, TypeError):
                    st.metric("VPA", "N/D")
                
                # Dividend Yield
                st.metric("Dividend Yield (12m)", "N/D")
                st.metric("Dividend Yield (Proj.)", "N/D")
            
            # Exibir resultados de valuation
            if valuation_result:
                st.subheader("Preço Justo Estimado")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Preço Atual", f"R$ {valuation_result['current_price']:.2f}")
                with col2:
                    st.metric("Preço Justo (Média)", f"R$ {valuation_result['mean']:.2f}", f"{valuation_result['upside_mean']:.1f}%")
                with col3:
                    st.metric("Preço Justo (Mediana)", f"R$ {valuation_result['median']:.2f}", f"{valuation_result['upside_median']:.1f}%")
                
                # Detalhes dos modelos
                st.subheader("Detalhes por Modelo de Valuation")
                model_data = []
                for model, price in valuation_result['models'].items():
                    upside = ((price / valuation_result['current_price']) - 1) * 100
                    model_data.append({
                        "Modelo": model,
                        "Preço Justo": f"R$ {price:.2f}",
                        "Upside/Downside": f"{upside:.1f}%"
                    })
                
                st.table(pd.DataFrame(model_data))
            
            # Recomendação baseada nos múltiplos e fase do ciclo
            st.subheader("Recomendação")
            recomendacao, justificativa = generate_recommendation(ticker_data, current_cycle_phase, valuation_result)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Recomendação:** {recomendacao}")
            with col2:
                st.info(f"**Justificativa:** {justificativa}")
        else:
            st.error(f"Dados fundamentalistas para {acao_selecionada_valuation} não disponíveis.")

    # --- Aba 3: Sinais de Market Timing ---
    with tab3:
        st.header("Sinais de Market Timing")
        st.markdown("Esta seção apresenta uma análise do ciclo econômico e alertas baseados nos indicadores macroeconômicos disponíveis.")
        
        # Fase do Ciclo
        st.subheader("Fase Estimada do Ciclo Econômico")
        
        # Exibir nível de confiança
        confianca = timing_signals.get('confianca_fase', 0)
        confianca_percent = int(confianca * 100)
        
        # Determinar cor e emoji com base na fase
        if "Expansão" in current_cycle_phase:
            emoji = "📈"
            color = "success"
        elif "Pico" in current_cycle_phase:
            emoji = "⚠️"
            color = "warning"
        elif "Contração" in current_cycle_phase:
            emoji = "📉"
            color = "error"
        elif "Recuperação" in current_cycle_phase:
            emoji = "🔄"
            color = "info"
        else:
            emoji = "🤔"
            color = "warning"
        
        # Exibir fase com barra de progresso para confiança
        st.markdown(f"**{current_cycle_phase}** {emoji}")
        st.progress(confianca)
        st.caption(f"Nível de confiança: {confianca_percent}%")
        
        # Explicação da fase
        if "Indefinida" in current_cycle_phase:
            st.markdown("Não foi possível determinar a fase do ciclo com confiança suficiente com base nos critérios atuais e dados disponíveis.")
            
            # Mostrar dados faltantes se houver
            if 'fase_debug' in st.session_state and 'missing_data' in st.session_state['fase_debug']:
                missing = st.session_state['fase_debug']['missing_data']
                if missing:
                    st.markdown(f"**Dados faltantes:** {', '.join(missing)}")
        elif "Provável" in current_cycle_phase or "Possível" in current_cycle_phase:
            st.markdown("Fase estimada com base em critérios parciais. Alguns indicadores sugerem esta fase, mas não há confirmação completa.")
        elif "Expansão" in current_cycle_phase:
            st.markdown("Economia em crescimento, inflação controlada, desemprego em queda e mercado em alta.")
        elif "Pico" in current_cycle_phase:
            st.markdown("Economia aquecida, inflação acelerando, juros subindo e mercado ainda resiliente.")
        elif "Contração" in current_cycle_phase:
            st.markdown("Economia desacelerando, inflação ainda alta, desemprego subindo e mercado em queda.")
        elif "Recuperação" in current_cycle_phase:
            st.markdown("Economia iniciando recuperação, inflação controlada, desemprego ainda alto mas mercado antecipando melhora.")
        
        # Mostrar dados de debug se solicitado
        if mostrar_debug and 'fase_debug' in st.session_state:
            st.subheader("Dados de Debug - Fase do Ciclo")
            st.json(st.session_state['fase_debug'])
        
        # Tendências dos Indicadores
        st.subheader("Tendências dos Indicadores")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tendência Selic", timing_signals.get('selic', 'N/D'))
        with col2:
            st.metric("Tendência IPCA", timing_signals.get('ipca', 'N/D'))
        with col3:
            st.metric("Tendência Desemprego", timing_signals.get('desemprego', 'N/D'))
        with col4:
            st.metric("Tendência Ibovespa", timing_signals.get('ibov', 'N/D'))
        
        # Alertas
        if timing_alerts:
            st.subheader("Alertas")
            for alert in timing_alerts:
                st.warning(alert)
        
        # Ciclos Econômicos Semelhantes
        if similar_cycles:
            st.subheader("Ciclos Econômicos Semelhantes ao Atual")
            st.markdown("Períodos históricos com características macroeconômicas semelhantes ao momento atual:")
            
            for i, cycle in enumerate(similar_cycles):
                similarity_percent = int(cycle['similarity'] * 100)
                st.markdown(f"**{i+1}. {cycle['period']}** (Similaridade: {similarity_percent}%)")
                
                # Mostrar características do período
                features = cycle['features']
                feature_text = []
                if features['selic'] == 1:
                    feature_text.append("Selic em alta")
                elif features['selic'] == -1:
                    feature_text.append("Selic em queda")
                
                if features['ipca'] == 1:
                    feature_text.append("Inflação acelerando")
                elif features['ipca'] == -1:
                    feature_text.append("Inflação desacelerando")
                
                if features['desemprego'] == 1:
                    feature_text.append("Desemprego subindo")
                elif features['desemprego'] == -1:
                    feature_text.append("Desemprego caindo")
                
                if features['ibov'] == 1:
                    feature_text.append("Ibovespa em alta")
                elif features['ibov'] == -1:
                    feature_text.append("Ibovespa em queda")
                
                st.caption(", ".join(feature_text))

    # --- Aba 4: Alocação Sugerida ---
    with tab4:
        st.header("Alocação Sugerida por Fase do Ciclo")
        st.markdown("Esta seção apresenta sugestões gerais de alocação com base na fase atual do ciclo econômico. As recomendações são simplificadas e devem ser adaptadas ao seu perfil de investimento e objetivos.")
        
        # Tabela de alocação sugerida por fase
        alocacao_data = {
            "Classe de Ativos": ["Ações", "Renda Fixa", "Imóveis", "Commodities", "Caixa"],
            "Expansão": ["Sobreponderado", "Subponderado", "Neutro", "Sobreponderado", "Subponderado"],
            "Pico": ["Neutro", "Neutro", "Sobreponderado", "Sobreponderado", "Neutro"],
            "Contração": ["Subponderado", "Sobreponderado", "Subponderado", "Neutro", "Sobreponderado"],
            "Recuperação": ["Sobreponderado", "Neutro", "Subponderado", "Subponderado", "Neutro"],
            "Indefinida": ["Neutro", "Neutro", "Neutro", "Neutro", "Neutro"]
        }
        
        df_alocacao = pd.DataFrame(alocacao_data)
        
        # Extrair a fase base (sem o "Provável", "Possível", etc.)
        base_phase = current_cycle_phase
        for prefix in ["Provável ", "Possível "]:
            if current_cycle_phase.startswith(prefix):
                base_phase = current_cycle_phase.replace(prefix, "")
                if "(" in base_phase:
                    base_phase = base_phase.split("(")[0].strip()
                break
        
        # Verificar se a fase base está nas colunas
        phase_in_columns = False
        for col in df_alocacao.columns:
            if col in base_phase:
                phase_in_columns = True
                base_phase = col
                break
        
        # Destacar a fase atual
        if phase_in_columns:
            st.markdown(f"**Fase atual (estimada): {current_cycle_phase}**")
            st.dataframe(df_alocacao[["Classe de Ativos", base_phase]])
        else:
            st.markdown("**Fase atual: Indefinida**")
            st.dataframe(df_alocacao[["Classe de Ativos", "Indefinida"]])
        
        st.markdown("### Visão Completa por Fase")
        st.dataframe(df_alocacao)
        
        st.markdown("""
        ### Interpretação:
        - **Sobreponderado**: Aumentar exposição acima do seu nível normal
        - **Neutro**: Manter exposição no seu nível normal
        - **Subponderado**: Reduzir exposição abaixo do seu nível normal
        
        ### Observações Importantes:
        - Estas são sugestões genéricas baseadas em comportamentos históricos dos mercados
        - Sua alocação pessoal deve considerar seu perfil de risco, horizonte de investimento e objetivos
        - Consulte um profissional certificado antes de tomar decisões de investimento
        """)
        
        # Setores recomendados por fase do ciclo
        st.subheader("Setores Recomendados por Fase do Ciclo")
        
        setores_data = {
            "Setor": ["Financeiro", "Tecnologia", "Consumo Cíclico", "Consumo Não-Cíclico", "Energia", "Materiais Básicos", "Saúde", "Utilidades"],
            "Expansão": ["Neutro", "Sobreponderado", "Sobreponderado", "Neutro", "Sobreponderado", "Sobreponderado", "Neutro", "Subponderado"],
            "Pico": ["Subponderado", "Neutro", "Neutro", "Sobreponderado", "Sobreponderado", "Neutro", "Sobreponderado", "Neutro"],
            "Contração": ["Subponderado", "Subponderado", "Subponderado", "Sobreponderado", "Neutro", "Subponderado", "Sobreponderado", "Sobreponderado"],
            "Recuperação": ["Sobreponderado", "Sobreponderado", "Neutro", "Neutro", "Neutro", "Sobreponderado", "Neutro", "Neutro"],
            "Indefinida": ["Neutro", "Neutro", "Neutro", "Neutro", "Neutro", "Neutro", "Neutro", "Neutro"]
        }
        
        df_setores = pd.DataFrame(setores_data)
        
        # Destacar a fase atual
        if phase_in_columns:
            st.dataframe(df_setores[["Setor", base_phase]])
        else:
            st.dataframe(df_setores[["Setor", "Indefinida"]])
        
        st.markdown("### Visão Completa por Fase")
        st.dataframe(df_setores)

    # --- Aba 5: Timing Histórico ---
    with tab5:
        st.header("Análise de Timing Histórico")
        st.markdown("Esta seção analisa o desempenho histórico de ações em ciclos econômicos semelhantes ao atual, ajudando a identificar oportunidades de timing de mercado.")
        
        # Mostrar tickers com melhor desempenho em ciclos semelhantes
        if best_performers:
            st.subheader("Ações com Melhor Desempenho em Ciclos Semelhantes")
            
            for i, period_data in enumerate(best_performers):
                st.markdown(f"### {i+1}. Período: {period_data['period']} (Similaridade: {int(period_data['similarity']*100)}%)")
                
                # Criar tabela de performers
                performers_data = []
                for perf in period_data['performers']:
                    performers_data.append({
                        "Ticker": perf['ticker'],
                        "Setor": perf['sector'],
                        "Retorno": f"{perf['performance']:.2f}%"
                    })
                
                st.table(pd.DataFrame(performers_data))
                
                # Adicionar gráfico de desempenho
                if performers_data:
                    # Extrair dados para gráfico
                    tickers = [p['ticker'] for p in period_data['performers']]
                    performances = [float(p['performance']) for p in period_data['performers']]
                    
                    # Criar gráfico de barras
                    fig = px.bar(
                        x=tickers,
                        y=performances,
                        labels={'x': 'Ticker', 'y': 'Retorno (%)'},
                        title=f"Desempenho no Período {period_data['period']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não foi possível identificar ciclos econômicos históricos suficientemente semelhantes ao atual para análise de desempenho.")
        
        # Análise de correlação entre ativos e indicadores macroeconômicos
        st.subheader("Correlação entre Ativos e Indicadores Macroeconômicos")
        
        # Selecionar ticker para análise detalhada
        ticker_analise = st.selectbox(
            "Escolha uma ação para análise detalhada",
            options=TICKERS,
            index=TICKERS.index(acao_selecionada_valuation) if acao_selecionada_valuation in TICKERS else 0
        )
        
        # Buscar dados históricos do ticker selecionado
        ticker_hist = historical_ticker_data.get(ticker_analise)
        
        if ticker_hist is not None and not ticker_hist.empty:
            # Preparar dados para análise de correlação
            try:
                # Resample para mensal para alinhar com dados macro
                ticker_monthly = ticker_hist['Close'].resample('M').last()
                
                # Criar DataFrame combinado
                corr_data = pd.DataFrame(ticker_monthly)
                corr_data.columns = [ticker_analise]
                
                # Adicionar dados macro
                if not df_selic_meta.empty:
                    corr_data['Selic'] = df_selic_meta.resample('M').last()
                
                if not df_ipca.empty:
                    corr_data['IPCA'] = df_ipca.resample('M').last()
                
                if not df_desemprego.empty:
                    corr_data['Desemprego'] = df_desemprego.resample('M').last()
                
                # Calcular correlação
                corr_matrix = corr_data.corr()
                
                # Exibir matriz de correlação
                st.markdown(f"#### Correlação de {ticker_analise} com Indicadores Macroeconômicos")
                
                # Criar heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title=f"Matriz de Correlação - {ticker_analise}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretação
                st.markdown("#### Interpretação da Correlação")
                
                # Extrair correlações com o ticker
                ticker_corrs = corr_matrix[ticker_analise].drop(ticker_analise)
                
                for indicator, corr in ticker_corrs.items():
                    if abs(corr) > 0.7:
                        strength = "forte"
                    elif abs(corr) > 0.3:
                        strength = "moderada"
                    else:
                        strength = "fraca"
                    
                    direction = "positiva" if corr > 0 else "negativa"
                    
                    st.markdown(f"- **{indicator}**: Correlação {strength} {direction} ({corr:.2f})")
                
                # Gráfico de preço vs. indicadores
                st.markdown("#### Evolução Histórica")
                
                # Normalizar dados para comparação
                scaler = MinMaxScaler()
                normalized_data = pd.DataFrame(
                    scaler.fit_transform(corr_data.dropna()),
                    columns=corr_data.columns,
                    index=corr_data.dropna().index
                )
                
                # Criar gráfico
                fig = go.Figure()
                
                # Adicionar linha para o ticker
                fig.add_trace(go.Scatter(
                    x=normalized_data.index,
                    y=normalized_data[ticker_analise],
                    mode='lines',
                    name=ticker_analise
                ))
                
                # Adicionar linhas para indicadores
                for col in normalized_data.columns:
                    if col != ticker_analise:
                        fig.add_trace(go.Scatter(
                            x=normalized_data.index,
                            y=normalized_data[col],
                            mode='lines',
                            name=col,
                            line=dict(dash='dash')
                        ))
                
                fig.update_layout(
                    title=f"Evolução Normalizada - {ticker_analise} vs. Indicadores",
                    xaxis_title="Data",
                    yaxis_title="Valor Normalizado (0-1)",
                    legend_title="Indicador"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro na análise de correlação: {e}")
        else:
            st.warning(f"Dados históricos insuficientes para {ticker_analise}.")
        
        # Mostrar dados de debug se solicitado
        if mostrar_debug and 'debug_info' in st.session_state:
            st.subheader("Dados de Debug - Disponibilidade de Dados")
            st.json(st.session_state['debug_info'])

if __name__ == "__main__":
    main()
