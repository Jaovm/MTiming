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

# --- Funções de Coleta de Dados Online ---
@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_bcb_data(series_id, series_name, start_date=None, end_date=None):
    """Busca dados de séries temporais do Banco Central do Brasil."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%d/%m/%Y')
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
def fetch_yahoo_finance_data(ticker, period="5y", interval="1d"):
    """Busca dados históricos de preços do Yahoo Finance."""
    # Adiciona sufixo .SA para ações brasileiras se não for um índice
    yahoo_ticker = ticker if ticker.startswith('^') else f"{ticker}.SA"
    
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
    
    try:
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
        st.warning(f"Erro ao buscar dados do Yahoo Finance para {ticker}: {e}")
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

    # 1. Tendência da Selic (últimos 3 meses)
    if not df_selic.empty and len(df_selic) >= 3:
        selic_trend = df_selic.iloc[-1].iloc[0] - df_selic.iloc[-4].iloc[0] if len(df_selic) >= 4 else 0
        if selic_trend > 0.5: signals['selic'] = 'Alta Forte'
        elif selic_trend > 0.1: signals['selic'] = 'Alta'
        elif selic_trend < -0.5: signals['selic'] = 'Queda Forte'
        elif selic_trend < -0.1: signals['selic'] = 'Queda'
        else: signals['selic'] = 'Estável'
        if signals['selic'] == 'Alta Forte': alerts.append("⚠️ Alerta: Juros subindo agressivamente!")
    else: signals['selic'] = 'N/D'

    # 2. Tendência da Inflação (IPCA - média móvel 3m vs 6m)
    if not df_ipca.empty and len(df_ipca) >= 6:
        ipca_3m = df_ipca.iloc[-3:].mean().iloc[0]
        ipca_6m = df_ipca.iloc[-6:].mean().iloc[0]
        if ipca_3m > ipca_6m + 0.1: signals['ipca'] = 'Acelerando'
        elif ipca_3m < ipca_6m - 0.1: signals['ipca'] = 'Desacelerando'
        else: signals['ipca'] = 'Estável'
        if signals['ipca'] == 'Acelerando': alerts.append("⚠️ Alerta: Inflação acelerando!")
    else: signals['ipca'] = 'N/D'

    # 3. Tendência do Desemprego (últimos 2 trimestres)
    if not df_desemprego.empty and len(df_desemprego) >= 2:
        desemprego_trend = df_desemprego.iloc[-1].iloc[0] - df_desemprego.iloc[-2].iloc[0]
        if desemprego_trend > 0.2: signals['desemprego'] = 'Subindo'
        elif desemprego_trend < -0.2: signals['desemprego'] = 'Caindo'
        else: signals['desemprego'] = 'Estável'
    else: signals['desemprego'] = 'N/D'

    # 4. Tendência do Ibovespa (média móvel 50d vs 200d)
    if not df_ibov.empty and len(df_ibov) >= 200:
        ibov_ma50 = df_ibov['Close'].rolling(window=50).mean().iloc[-1]
        ibov_ma200 = df_ibov['Close'].rolling(window=200).mean().iloc[-1]
        if ibov_ma50 > ibov_ma200 * 1.02: signals['ibov'] = 'Alta (MM50 > MM200)'
        elif ibov_ma50 < ibov_ma200 * 0.98: signals['ibov'] = 'Queda (MM50 < MM200)'
        else: signals['ibov'] = 'Lateral'
    else: signals['ibov'] = 'N/D'

    # 5. Determinar Fase do Ciclo (Lógica Melhorada)
    fase = "Indefinida"
    
    # Verificar se temos dados suficientes para determinar a fase
    dados_suficientes = all(signals.get(key) != 'N/D' for key in ['selic', 'ipca', 'desemprego', 'ibov'])
    
    if not dados_suficientes:
        fase = "Indefinida (Dados Insuficientes)"
    elif signals.get('selic') in ['Queda', 'Queda Forte', 'Estável'] and \
       signals.get('ipca') in ['Desacelerando', 'Estável'] and \
       signals.get('desemprego') in ['Caindo', 'Estável'] and \
       signals.get('ibov') == 'Alta (MM50 > MM200)':
        fase = "Expansão"
    elif signals.get('selic') in ['Alta', 'Alta Forte'] and \
         signals.get('ipca') == 'Acelerando' and \
         signals.get('desemprego') in ['Estável', 'Caindo'] and \
         signals.get('ibov') in ['Lateral', 'Alta (MM50 > MM200)']:
        fase = "Pico"
    elif signals.get('selic') in ['Alta', 'Alta Forte', 'Estável'] and \
         signals.get('ipca') in ['Estável', 'Acelerando'] and \
         signals.get('desemprego') == 'Subindo' and \
         signals.get('ibov') == 'Queda (MM50 < MM200)':
        fase = "Contração"
    elif signals.get('selic') in ['Queda', 'Queda Forte'] and \
         signals.get('ipca') in ['Desacelerando', 'Estável'] and \
         signals.get('desemprego') in ['Subindo', 'Estável'] and \
         signals.get('ibov') in ['Lateral', 'Queda (MM50 < MM200)']:
        fase = "Recuperação"
    # Regra mais flexível para determinar fase quando não se encaixa perfeitamente
    elif signals.get('selic') in ['Queda', 'Queda Forte'] and signals.get('ibov') == 'Alta (MM50 > MM200)':
        fase = "Provável Expansão"
    elif signals.get('selic') in ['Alta', 'Alta Forte'] and signals.get('ipca') == 'Acelerando':
        fase = "Provável Pico"
    elif signals.get('desemprego') == 'Subindo' and signals.get('ibov') == 'Queda (MM50 < MM200)':
        fase = "Provável Contração"
    elif signals.get('selic') in ['Queda', 'Queda Forte'] and signals.get('ipca') in ['Desacelerando', 'Estável']:
        fase = "Provável Recuperação"

    signals['fase_ciclo'] = fase

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

# --- Função para Gerar Recomendações ---
def generate_recommendation(ticker_data, cycle_phase):
    """Gera recomendação com base nos múltiplos atuais e fase do ciclo."""
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
    if pl is None or pvp is None:
        recomendacao = "N/D"
        justificativa = f"Dados de múltiplos (P/L={pl_display}, P/VP={pvp_display}) indisponíveis ou inválidos. Fase: {cycle_phase}."
        return recomendacao, justificativa # Retorna imediatamente

    # Lógica baseada na fase e valuation (agora sabemos que pl e pvp não são None)
    if cycle_phase == "Expansão" or cycle_phase == "Provável Expansão":
        if val_pl == "Baixo" and val_pvp == "Baixo": recomendacao = "Compra Forte"
        elif val_pl != "Alto" and val_pvp != "Alto": recomendacao = "Compra"
        else: recomendacao = "Neutro"
    elif cycle_phase == "Pico" or cycle_phase == "Provável Pico":
        if val_pl == "Alto" or val_pvp == "Alto": recomendacao = "Venda"
        elif val_pl == "Médio" and val_pvp == "Médio": recomendacao = "Neutro/Venda"
        else: recomendacao = "Neutro"
    elif cycle_phase == "Contração" or cycle_phase == "Provável Contração":
        if val_pl == "Alto" or val_pvp == "Alto": recomendacao = "Venda Forte"
        elif val_pl == "Médio" or val_pvp == "Médio": recomendacao = "Venda"
        else: recomendacao = "Neutro/Avaliar"
    elif cycle_phase == "Recuperação" or cycle_phase == "Provável Recuperação":
        if val_pl == "Baixo" and val_pvp == "Baixo": recomendacao = "Compra"
        elif val_pl != "Alto" and val_pvp != "Alto": recomendacao = "Compra/Neutro"
        else: recomendacao = "Neutro"
    # Se fase indefinida, recomendação é Neutro, mas a justificativa já foi montada
    elif "Indefinida" in cycle_phase:
         recomendacao = "Neutro"
         # Ajusta a justificativa para ser mais clara sobre a fase
         justificativa = f"Fase do ciclo {cycle_phase}. P/L: {pl_display} ({val_pl}), P/VP: {pvp_display} ({val_pvp})."

    # Atualiza a justificativa final caso não tenha caído no caso de N/D inicial
    if recomendacao != "N/D":
        justificativa = f"P/L: {pl_display} ({val_pl}), P/VP: {pvp_display} ({val_pvp}). Fase do ciclo: {cycle_phase}. Recomendação: {recomendacao}."

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
        ibov_df = fetch_yahoo_finance_data("^BVSP")
        if not ibov_df.empty:
            macro_data["ibovespa"] = ibov_df
        
        # Coletar VIX
        vix_df = fetch_yahoo_finance_data("^VIX")
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
    if st.sidebar.button("Atualizar Dados"):
        st.cache_data.clear()
        st.experimental_rerun()
    last_update_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    st.sidebar.info(f"Última atualização: {last_update_time}")
    st.sidebar.caption("Desenvolvido por Manus")

    # --- Carregamento de Dados ---
    macro_data = load_all_macro_data()
    fundamental_data = load_all_fundamental_data()
    
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

    # --- Área Principal (Abas) ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cenário Macroeconômico",
        "📈 Valuation (Atual)",
        "🚦 Sinais de Market Timing",
        "🧭 Alocação Sugerida"
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
                last_ibov = df_ibov['Close'].iloc[-1]
                prev_ibov = df_ibov['Close'].iloc[-2] if len(df_ibov) > 1 else last_ibov
                delta_ibov = ((last_ibov / prev_ibov) - 1) * 100 if prev_ibov != 0 else 0
                st.metric("Ibovespa (Fechamento)", f"{last_ibov:,.0f}", f"{delta_ibov:.2f}%", delta_color="normal", help="Último valor de fechamento do Índice Bovespa.")
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
        st.markdown("Atenção: Esta aba exibe os indicadores fundamentalistas atuais da ação selecionada. Devido a limitações na fonte de dados gratuita, não foi possível incluir a análise histórica dos múltiplos (P/L, P/VP, etc.).")
        
        ticker_data = fundamental_data.get(acao_selecionada_valuation)
        
        if ticker_data:
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
            
            # Recomendação baseada nos múltiplos e fase do ciclo
            st.subheader("Recomendação")
            recomendacao, justificativa = generate_recommendation(ticker_data, current_cycle_phase)
            
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
        st.markdown("Esta seção apresenta uma análise simplificada do ciclo econômico e alertas baseados nos indicadores macroeconômicos disponíveis.")
        
        # Fase do Ciclo
        st.subheader("Fase Estimada do Ciclo Econômico")
        if "Indefinida" in current_cycle_phase:
            st.warning(f"{current_cycle_phase} 🤔")
            st.markdown("Não foi possível determinar a fase do ciclo com base nos critérios atuais e dados disponíveis.")
        elif "Provável" in current_cycle_phase:
            st.info(f"{current_cycle_phase} 🔍")
            st.markdown("Fase estimada com base em critérios parciais. Alguns indicadores sugerem esta fase, mas não há confirmação completa.")
        elif current_cycle_phase == "Expansão":
            st.success(f"{current_cycle_phase} 📈")
            st.markdown("Economia em crescimento, inflação controlada, desemprego em queda e mercado em alta.")
        elif current_cycle_phase == "Pico":
            st.warning(f"{current_cycle_phase} ⚠️")
            st.markdown("Economia aquecida, inflação acelerando, juros subindo e mercado ainda resiliente.")
        elif current_cycle_phase == "Contração":
            st.error(f"{current_cycle_phase} 📉")
            st.markdown("Economia desacelerando, inflação ainda alta, desemprego subindo e mercado em queda.")
        elif current_cycle_phase == "Recuperação":
            st.info(f"{current_cycle_phase} 🔄")
            st.markdown("Economia iniciando recuperação, inflação controlada, desemprego ainda alto mas mercado antecipando melhora.")
        
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
        
        # Destacar a fase atual
        if "Provável" in current_cycle_phase:
            # Extrai a fase base (sem o "Provável")
            base_phase = current_cycle_phase.replace("Provável ", "")
            if base_phase in df_alocacao.columns:
                st.markdown(f"**Fase atual (estimada): {current_cycle_phase}**")
                st.dataframe(df_alocacao[["Classe de Ativos", base_phase]])
            else:
                st.markdown("**Fase atual: Indefinida**")
                st.dataframe(df_alocacao[["Classe de Ativos", "Indefinida"]])
        elif current_cycle_phase in df_alocacao.columns:
            st.markdown(f"**Fase atual: {current_cycle_phase}**")
            st.dataframe(df_alocacao[["Classe de Ativos", current_cycle_phase]])
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

if __name__ == "__main__":
    main()
