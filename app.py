# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
import numpy as np # Para cálculos numéricos

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel de Market Timing Brasil",
    page_icon="📊",
    layout="wide"
)

# --- Diretórios de Dados ---
DATA_DIR = "."
FUNDAMENTAL_DATA_DIR = "."

# --- Lista de Tickers (consistente com a coleta) ---
TICKERS = [
    "AGRO3", "BBAS3", "BBSE3", "BPAC11", "EGIE3",
    "ITUB3", "PRIO3", "PSSA3", "SAPR3", "SBSP3",
    "VIVT3", "WEGE3", "TOTS3", "B3SA3", "TAEE3",
    "CMIG3"
]

# --- Funções de Carregamento e Processamento de Dados ---
@st.cache_data
def load_data(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            # Verificar se existe coluna 'data' ou similar
            possible_date_cols = [col for col in df.columns if col.lower() in ['data', 'date', 'Data', 'Date']]
            if possible_date_cols:
                df = df.set_index(possible_date_cols[0])
                df.index = pd.to_datetime(df.index, errors='coerce')
            else:
                st.error(f"O arquivo {file_name} não possui coluna de data.")
                return pd.DataFrame()

            # Tratamento de colunas
            if len(df.columns) == 1 and df.columns[0].lower() == 'valor':
                df.columns = [file_name.replace('_data.csv', '')]
            if 'Close' in df.columns:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            if len(df.columns) == 1:
                df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors='coerce')

            df = df.sort_index()
            df = df.dropna()

            return df
        except Exception as e:
            st.error(f"Erro ao carregar {file_name}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Arquivo não encontrado: {file_name}")
        return pd.DataFrame()
        

@st.cache_data
def load_fundamental_snapshot(ticker):
    """Carrega o arquivo JSON de snapshot fundamental para um ticker."""
    file_path = os.path.join(FUNDAMENTAL_DATA_DIR, f"{ticker}_fundamental_snapshot.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Erro ao carregar JSON fundamental para {ticker}: {e}")
            return None
    else:
        print(f"Arquivo snapshot não encontrado para {ticker}: {file_path}")
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

# --- Função para Calcular Sinais de Market Timing (Lógica Simples Inicial) ---
def calculate_market_timing_signals(df_selic, df_ipca, df_desemprego, df_ibov):
    """Calcula sinais simples de market timing com base nas tendências recentes."""
    signals = {}
    alerts = []

    # 1. Tendência da Selic (últimos 3 meses)
    if not df_selic.empty and len(df_selic) >= 3:
        selic_trend = df_selic[df_selic.columns[0]].iloc[-1] - df_selic[df_selic.columns[0]].iloc[-4]
        if selic_trend > 0.5: signals['selic'] = 'Alta Forte'
        elif selic_trend > 0.1: signals['selic'] = 'Alta'
        elif selic_trend < -0.5: signals['selic'] = 'Queda Forte'
        elif selic_trend < -0.1: signals['selic'] = 'Queda'
        else: signals['selic'] = 'Estável'
        if signals['selic'] == 'Alta Forte': alerts.append("⚠️ Alerta: Juros subindo agressivamente!")
    else: signals['selic'] = 'N/D'

    # 2. Tendência da Inflação (IPCA - média móvel 3m vs 6m)
    if not df_ipca.empty and len(df_ipca) >= 6:
        ipca_3m = df_ipca[df_ipca.columns[0]].rolling(window=3).mean().iloc[-1]
        ipca_6m = df_ipca[df_ipca.columns[0]].rolling(window=6).mean().iloc[-1]
        if ipca_3m > ipca_6m + 0.1: signals['ipca'] = 'Acelerando'
        elif ipca_3m < ipca_6m - 0.1: signals['ipca'] = 'Desacelerando'
        else: signals['ipca'] = 'Estável'
        if signals['ipca'] == 'Acelerando': alerts.append("⚠️ Alerta: Inflação acelerando!")
    else: signals['ipca'] = 'N/D'

    # 3. Tendência do Desemprego (últimos 2 trimestres)
    if not df_desemprego.empty and len(df_desemprego) >= 2:
        desemprego_trend = df_desemprego[df_desemprego.columns[0]].iloc[-1] - df_desemprego[df_desemprego.columns[0]].iloc[-2]
        if desemprego_trend > 0.2: signals['desemprego'] = 'Subindo'
        elif desemprego_trend < -0.2: signals['desemprego'] = 'Caindo'
        else: signals['desemprego'] = 'Estável'
    else: signals['desemprego'] = 'N/D'

    # 4. Tendência do Ibovespa (média móvel 50d vs 200d)
    if not df_ibov.empty and len(df_ibov) >= 200 and 'Close' in df_ibov.columns:
        ibov_ma50 = df_ibov['Close'].rolling(window=50).mean().iloc[-1]
        ibov_ma200 = df_ibov['Close'].rolling(window=200).mean().iloc[-1]
        if ibov_ma50 > ibov_ma200 * 1.02: signals['ibov'] = 'Alta (MM50 > MM200)'
        elif ibov_ma50 < ibov_ma200 * 0.98: signals['ibov'] = 'Queda (MM50 < MM200)'
        else: signals['ibov'] = 'Lateral'
    else: signals['ibov'] = 'N/D'

    # 5. Determinar Fase do Ciclo (Lógica MUITO Simplificada - EXEMPLO)
    fase = "Indefinida"
    if signals.get('selic') in ['Queda', 'Queda Forte', 'Estável'] and \
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

    signals['fase_ciclo'] = fase

    # Adicionar VIX (se disponível)
    if not df_vix.empty and 'Close' in df_vix.columns:
        last_vix = df_vix['Close'].iloc[-1]
        signals['vix'] = f"{last_vix:.2f}"
        if last_vix > 30: # Exemplo de limite para alerta
            alerts.append(f"⚠️ Alerta: VIX alto ({last_vix:.2f})!")
    else:
        signals['vix'] = 'N/D'

    return signals, alerts

# --- Função para Gerar Recomendações (Lógica Simples Inicial) ---
def generate_recommendation(ticker_data, cycle_phase):
    """Gera recomendação com base nos múltiplos atuais e fase do ciclo."""
    if not ticker_data:
        return "N/D", "Dados fundamentalistas indisponíveis."

    # Extrair múltiplos (tratar possíveis Nones ou erros de conversão)
    try:
        pl = float(ticker_data.get("trailingPE", 999))
    except (ValueError, TypeError):
        pl = 999 # Valor alto para indicar problema ou ausência
    try:
        pvp = float(ticker_data.get("priceToBook", 999))
    except (ValueError, TypeError):
        pvp = 999

    # Definir Níveis de Valuation (Exemplo Simplificado)
    val_pl = "Alto" if pl > 20 else ("Médio" if pl > 10 else "Baixo")
    val_pvp = "Alto" if pvp > 2 else ("Médio" if pvp > 1 else "Baixo")

    # Lógica de Recomendação
    recomendacao = "Neutro"
    justificativa = f"P/L: {pl:.1f} ({val_pl}), P/VP: {pvp:.1f} ({val_pvp}) em fase de {cycle_phase}."

    if cycle_phase == "Expansão":
        if val_pl == "Baixo" and val_pvp == "Baixo": recomendacao = "Compra Forte"
        elif val_pl != "Alto" and val_pvp != "Alto": recomendacao = "Compra"
        else: recomendacao = "Neutro"
    elif cycle_phase == "Pico":
        if val_pl == "Alto" or val_pvp == "Alto": recomendacao = "Venda"
        elif val_pl == "Médio" and val_pvp == "Médio": recomendacao = "Neutro/Venda"
        else: recomendacao = "Neutro"
    elif cycle_phase == "Contração":
        if val_pl == "Alto" or val_pvp == "Alto": recomendacao = "Venda Forte"
        elif val_pl == "Médio" or val_pvp == "Médio": recomendacao = "Venda"
        else: recomendacao = "Neutro/Avaliar"
    elif cycle_phase == "Recuperação":
        if val_pl == "Baixo" and val_pvp == "Baixo": recomendacao = "Compra"
        elif val_pl != "Alto" and val_pvp != "Alto": recomendacao = "Compra/Neutro"
        else: recomendacao = "Neutro"
    # Se fase indefinida, recomendação é Neutro
    elif cycle_phase == "Indefinida":
         recomendacao = "Neutro"
         justificativa = f"Fase do ciclo indefinida. P/L: {pl:.1f}, P/VP: {pvp:.1f}."

    # Ajuste final para casos sem dados
    if pl == 999 or pvp == 999:
        recomendacao = "N/D"
        justificativa = f"Dados de múltiplos (P/L ou P/VP) indisponíveis ou inválidos. Fase: {cycle_phase}."


    return recomendacao, justificativa

# --- Carregamento Inicial dos Dados ---
# Macro/Mercado
df_ipca = load_data("ipca_data.csv")
df_igpm = load_data("igp-m_data.csv")
df_selic_meta = load_data("selic_meta_data.csv")
df_selic_efetiva = load_data("selic_efetiva_data.csv")
df_desemprego = load_data("desemprego_pnad_data.csv")
df_ibov = load_data("ibov_data.csv")
df_ibc_br = load_data("ibc-br_data.csv")
df_vix = load_data("vix_data.csv")

# Fundamental Snapshot
fundamental_snapshots = {}
for ticker in TICKERS:
    data = load_fundamental_snapshot(ticker)
    if data:
        fundamental_snapshots[ticker] = data

# Calcula sinais de timing uma vez
timing_signals, timing_alerts = calculate_market_timing_signals(df_selic_meta, df_ipca, df_desemprego, df_ibov)
current_cycle_phase = timing_signals.get('fase_ciclo', 'Indefinida')

# --- Barra Lateral (Sidebar) --- (Código Omitido para Brevidade - Igual ao anterior)
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
st.sidebar.info(f"Última atualização (simulada): {last_update_time}")
st.sidebar.caption("Desenvolvido por Manus")

# --- Área Principal (Abas) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Cenário Macroeconômico",
    "📈 Valuation (Atual)",
    "🚦 Sinais de Market Timing",
    "🧭 Alocação Sugerida"
])

# --- Aba 1: Cenário Macroeconômico --- (Código Omitido para Brevidade)
with tab1:
    st.header("Cenário Macroeconômico - Brasil")
    st.subheader("Indicadores Chave")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not df_ipca.empty:
            last_ipca = df_ipca.iloc[-1].iloc[0]
            prev_ipca = df_ipca.iloc[-2].iloc[0]
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
            prev_igpm = df_igpm.iloc[-2].iloc[0]
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
        if not df_ibov.empty and 'Close' in df_ibov.columns:
             last_ibov = df_ibov['Close'].iloc[-1]
             prev_ibov = df_ibov['Close'].iloc[-2]
             delta_ibov = ((last_ibov / prev_ibov) - 1) * 100 if prev_ibov != 0 else 0
             st.metric("Ibovespa (Fechamento)", f"{last_ibov:,.0f}", f"{delta_ibov:.2f}%", delta_color="normal", help="Último valor de fechamento do Índice Bovespa.")
        else:
             st.metric("Ibovespa (Fechamento)", "N/D")
    with col4:
        if not df_ibc_br.empty:
             last_ibc = df_ibc_br.iloc[-1].iloc[0]
             st.metric("IBC-Br (Proxy PIB)", f"{last_ibc:.2f}", help="Índice de Atividade Econômica do Banco Central.")
        else:
             st.metric("IBC-Br (Proxy PIB)", "N/D", help="Falha na coleta de dados.")
        if not df_vix.empty and 'Close' in df_vix.columns:
             last_vix = df_vix['Close'].iloc[-1]
             st.metric("VIX", f"{last_vix:.2f}", help="Índice de Volatilidade CBOE.")
        else:
             st.metric("VIX", "N/D", help="Falha na coleta de dados.")
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
    if not df_desemprego_hist.empty:
        fig_desemprego = px.line(df_desemprego_hist, y=df_desemprego_hist.columns[0], title='Taxa de Desemprego PNAD (%)')
        fig_desemprego.update_layout(yaxis_title='%', legend_title="Taxa")
        st.plotly_chart(fig_desemprego, use_container_width=True)
    else:
        st.write("Dados de desemprego não disponíveis para o período.")
    if not df_ibov_hist.empty and 'Close' in df_ibov_hist.columns:
        fig_ibov = px.line(df_ibov_hist, y='Close', title='Ibovespa (Pontos)')
        fig_ibov.update_layout(yaxis_title='Pontos', legend_title="Índice")
        st.plotly_chart(fig_ibov, use_container_width=True)
    else:
        st.write("Dados do Ibovespa não disponíveis para o período.")

# --- Aba 2: Valuation (Atual) --- (Código Omitido para Brevidade)
with tab2:
    st.header(f"Valuation Atual - {acao_selecionada_valuation}")
    st.info("Atenção: Esta aba exibe os indicadores fundamentalistas *atuais* da ação selecionada. Devido a limitações na fonte de dados gratuita, não foi possível incluir a análise histórica dos múltiplos (P/L, P/VP, etc.).")
    dados_acao = fundamental_snapshots.get(acao_selecionada_valuation)
    if dados_acao:
        st.subheader("Indicadores Principais")
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        def display_metric(column, key, label, help_text="", format_spec=":.2f"):
            value = dados_acao.get(key)
            if isinstance(value, str):
                try: value = float(value.replace('.', '').replace(',', '.'))
                except: pass
            if value is not None and isinstance(value, (int, float)):
                column.metric(label, f"{value:{format_spec}}", help=help_text)
            elif value is not None:
                 column.metric(label, str(value), help=help_text)
            else:
                column.metric(label, "N/D", help=help_text)
        with col_v1:
            display_metric(col_v1, "regularMarketPrice", "Preço Atual", "Preço de fechamento mais recente.", ":,.2f")
            display_metric(col_v1, "marketCap", "Valor de Mercado", "Capitalização de mercado em BRL.", ":,.0f")
        with col_v2:
            display_metric(col_v2, "trailingPE", "P/L (12m)", "Preço / Lucro por Ação (últimos 12 meses).")
            display_metric(col_v2, "forwardPE", "P/L (Proj.)", "Preço / Lucro por Ação (projetado).")
        with col_v3:
            display_metric(col_v3, "priceToBook", "P/VP", "Preço / Valor Patrimonial por Ação.")
            display_metric(col_v3, "bookValue", "VPA", "Valor Patrimonial por Ação.", ":,.2f")
        with col_v4:
            display_metric(col_v4, "trailingAnnualDividendYield", "Dividend Yield (12m)", "Dividendos pagos nos últimos 12 meses / Preço.", ":.2%")
            display_metric(col_v4, "dividendYield", "Dividend Yield (Proj.)", "Dividendos projetados / Preço.", ":.2%")
        st.divider()
        st.subheader("Detalhes Adicionais")
        summary_profile = dados_acao.get("summaryProfile")
        if summary_profile and isinstance(summary_profile, dict):
            st.markdown(f"**Setor:** {summary_profile.get('sector', 'N/D')}")
            st.markdown(f"**Indústria:** {summary_profile.get('industry', 'N/D')}")
            st.markdown(f"**Website:** {summary_profile.get('website', 'N/D')}")
            st.markdown(f"**Resumo:**")
            st.caption(summary_profile.get('longBusinessSummary', 'N/D'))
        else:
            st.markdown(f"**Nome:** {dados_acao.get('longName', 'N/D')}")
            st.markdown(f"**Símbolo:** {dados_acao.get('symbol', 'N/D')}")
            st.write("Dados detalhados de perfil da empresa (summaryProfile) não disponíveis.")
    else:
        st.error(f"Não foi possível carregar os dados fundamentalistas para {acao_selecionada_valuation}.")

# --- Aba 3: Sinais de Market Timing --- (Código Omitido para Brevidade)
with tab3:
    st.header("Sinais de Market Timing")
    st.info("Esta seção apresenta uma análise simplificada do ciclo econômico e alertas baseados nos indicadores macroeconômicos disponíveis.")
    # Exibe a Fase do Ciclo Estimada
    st.subheader("Fase Estimada do Ciclo Econômico")
    fase_ciclo = current_cycle_phase # Usa a variável calculada globalmente
    if fase_ciclo == "Expansão":
        st.success(f"**{fase_ciclo}** 🚀")
        st.caption("Indicadores sugerem crescimento econômico, juros baixos/estáveis, inflação controlada e mercado de ações em alta.")
    elif fase_ciclo == "Pico":
        st.warning(f"**{fase_ciclo}** ⛰️")
        st.caption("Indicadores sugerem atividade econômica forte, mas com pressões inflacionárias e juros em alta, podendo anteceder uma desaceleração.")
    elif fase_ciclo == "Contração":
        st.error(f"**{fase_ciclo}** 📉")
        st.caption("Indicadores sugerem desaceleração econômica, juros altos, inflação cedendo (ou ainda alta) e mercado de ações em baixa.")
    elif fase_ciclo == "Recuperação":
        st.info(f"**{fase_ciclo}** 🌱")
        st.caption("Indicadores sugerem fundo do ciclo, com juros começando a cair, inflação controlada e mercado de ações buscando recuperação.")
    else:
        st.info(f"**{fase_ciclo}** 🤔")
        st.caption("Não foi possível determinar a fase do ciclo com base nos critérios atuais e dados disponíveis.")
    st.divider()
    st.subheader("Tendências dos Indicadores")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Tendência Selic", timing_signals.get('selic', 'N/D'))
    col_s2.metric("Tendência IPCA", timing_signals.get('ipca', 'N/D'))
    col_s3.metric("Tendência Desemprego", timing_signals.get('desemprego', 'N/D'))
    col_s4.metric("Tendência Ibovespa", timing_signals.get('ibov', 'N/D'))
    st.divider()
    st.subheader("Alertas")
    if timing_alerts:
        for alert in timing_alerts:
            if "VIX" in alert: st.warning(alert)
            elif "Juros" in alert: st.error(alert)
            elif "Inflação" in alert: st.error(alert)
            else: st.info(alert)
    else:
        st.success("Nenhum alerta crítico identificado no momento.")
    st.caption("Nota: A análise de ciclo e os alertas são baseados em regras simplificadas e nos dados disponíveis. A inversão da curva de juros não pôde ser avaliada.")

# --- Aba 4: Alocação Sugerida ---
with tab4:
    st.header("Alocação Sugerida")
    st.info(f"Recomendações baseadas na fase atual do ciclo estimada: **{current_cycle_phase}** e nos múltiplos *atuais* das ações.")
    st.caption("Lembre-se: Esta é uma análise automatizada e simplificada. Faça sua própria análise antes de tomar decisões de investimento.")

    recomendacoes = []
    for ticker in TICKERS:
        dados_acao = fundamental_snapshots.get(ticker)
        recomendacao, justificativa = generate_recommendation(dados_acao, current_cycle_phase)
        recomendacoes.append({
            "Ação": ticker,
            "Recomendação": recomendacao,
            "Justificativa (Simplificada)": justificativa
        })

    if recomendacoes:
        df_recomendacoes = pd.DataFrame(recomendacoes)

        # Colorir células de recomendação
        def color_recomendacao(val):
            color = 'grey'
            if 'Compra Forte' in val: color = 'darkgreen'
            elif 'Compra' in val: color = 'lightgreen'
            elif 'Venda Forte' in val: color = 'darkred'
            elif 'Venda' in val: color = 'lightcoral'
            elif 'Neutro' in val: color = 'lightgrey'
            elif 'N/D' in val: color = 'white'
            return f'background-color: {color}'

        st.dataframe(
            df_recomendacoes.style.applymap(color_recomendacao, subset=['Recomendação']),
            use_container_width=True
        )
    else:
        st.warning("Não foi possível gerar recomendações.")

    st.caption("Critérios Simplificados Usados:")
    st.caption("- P/L: Baixo (<10), Médio (10-20), Alto (>20)")
    st.caption("- P/VP: Baixo (<1), Médio (1-2), Alto (>2)")
    st.caption("- A lógica combina a fase do ciclo com os níveis de P/L e P/VP atuais.")

