# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel de Market Timing Brasil",
    page_icon="📊",
    layout="wide"
)

# --- Diretórios de Dados ---
DATA_DIR = "."
FUNDAMENTAL_DATA_DIR = "."

# --- Lista de Tickers ---
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
            df = pd.read_csv(file_path, index_col="data", parse_dates=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if len(df.columns) == 1 and df.columns[0] == 'valor':
                df.columns = [file_name.replace('_data.csv', '')]
            if len(df.columns) == 1:
                df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors='coerce')
            elif 'Close' in df.columns:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(inplace=True)
            return df
        except Exception as e:
            st.error(f"Erro ao carregar {file_name}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

@st.cache_data
def load_fundamental_snapshot(ticker):
    file_path = os.path.join(FUNDAMENTAL_DATA_DIR, f"{ticker}_fundamental_snapshot.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"Erro ao carregar JSON para {ticker}: {e}")
            return None
    else:
        return None

def filter_data_by_period(df, period_option):
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

# --- Dados Macro e Mercado ---
df_selic = load_data("selic_data.csv")
df_ipca = load_data("ipca_data.csv")
df_desemprego = load_data("desemprego_data.csv")
df_ibov = load_data("ibov_data.csv")
df_vix = load_data("vix_data.csv")

# --- Sinais de Market Timing ---
def calculate_market_timing_signals():
    signals = {}
    alerts = []

    if not df_selic.empty:
        selic_trend = df_selic.iloc[-1, 0] - df_selic.iloc[-4, 0]
        if selic_trend > 0.5:
            signals['Selic'] = "Alta Forte"
            alerts.append("⚠️ Juros subindo agressivamente!")
        elif selic_trend > 0.1:
            signals['Selic'] = "Alta"
        elif selic_trend < -0.5:
            signals['Selic'] = "Queda Forte"
        elif selic_trend < -0.1:
            signals['Selic'] = "Queda"
        else:
            signals['Selic'] = "Estável"
    else:
        signals['Selic'] = "N/D"

    if not df_ipca.empty:
        ipca_3m = df_ipca.iloc[-3:, 0].mean()
        ipca_6m = df_ipca.iloc[-6:, 0].mean()
        if ipca_3m > ipca_6m + 0.1:
            signals['Inflação'] = "Acelerando"
            alerts.append("⚠️ Inflação acelerando!")
        elif ipca_3m < ipca_6m - 0.1:
            signals['Inflação'] = "Desacelerando"
        else:
            signals['Inflação'] = "Estável"
    else:
        signals['Inflação'] = "N/D"

    if not df_desemprego.empty:
        desemprego_trend = df_desemprego.iloc[-1, 0] - df_desemprego.iloc[-2, 0]
        if desemprego_trend > 0.2:
            signals['Desemprego'] = "Subindo"
        elif desemprego_trend < -0.2:
            signals['Desemprego'] = "Caindo"
        else:
            signals['Desemprego'] = "Estável"
    else:
        signals['Desemprego'] = "N/D"

    if not df_ibov.empty and 'Close' in df_ibov.columns:
        ma50 = df_ibov['Close'].rolling(50).mean().iloc[-1]
        ma200 = df_ibov['Close'].rolling(200).mean().iloc[-1]
        if ma50 > ma200 * 1.02:
            signals['Ibovespa'] = "Alta (MM50 > MM200)"
        elif ma50 < ma200 * 0.98:
            signals['Ibovespa'] = "Queda (MM50 < MM200)"
        else:
            signals['Ibovespa'] = "Lateral"
    else:
        signals['Ibovespa'] = "N/D"

    if not df_vix.empty and 'Close' in df_vix.columns:
        vix_last = df_vix['Close'].iloc[-1]
        signals['VIX'] = f"{vix_last:.2f}"
        if vix_last > 30:
            alerts.append(f"⚠️ VIX elevado ({vix_last:.2f})!")
    else:
        signals['VIX'] = "N/D"

    # Determinar fase do ciclo
    if signals.get('Selic') in ["Queda", "Queda Forte"] and \
       signals.get('Inflação') in ["Desacelerando", "Estável"] and \
       signals.get('Desemprego') in ["Caindo", "Estável"] and \
       signals.get('Ibovespa') == "Alta (MM50 > MM200)":
        ciclo = "Expansão"
    elif signals.get('Selic') in ["Alta", "Alta Forte"] and \
         signals.get('Inflação') == "Acelerando":
        ciclo = "Pico"
    elif signals.get('Desemprego') == "Subindo" and \
         signals.get('Ibovespa') == "Queda (MM50 < MM200)":
        ciclo = "Contração"
    elif signals.get('Selic') in ["Queda", "Queda Forte"] and \
         signals.get('Ibovespa') in ["Queda (MM50 < MM200)", "Lateral"]:
        ciclo = "Recuperação"
    else:
        ciclo = "Indefinido"

    signals['Ciclo Econômico'] = ciclo

    return signals, alerts

signals, alerts = calculate_market_timing_signals()

# --- Layout ---
st.title("📊 Painel de Market Timing Brasil")

st.subheader("Sinais de Market Timing")
st.write(signals)
if alerts:
    st.warning("\n".join(alerts))

st.subheader("Gráficos Macro e Mercado")
period = st.selectbox("Período", ["1A", "3A", "5A", "10A"])

col1, col2 = st.columns(2)

with col1:
    if not df_ibov.empty:
        df = filter_data_by_period(df_ibov, period)
        fig = px.line(df, y='Close', title="Ibovespa")
        st.plotly_chart(fig, use_container_width=True)

    if not df_selic.empty:
        df = filter_data_by_period(df_selic, period)
        fig = px.line(df, y=df.columns[0], title="Selic")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if not df_ipca.empty:
        df = filter_data_by_period(df_ipca, period)
        fig = px.line(df, y=df.columns[0], title="Inflação (IPCA)")
        st.plotly_chart(fig, use_container_width=True)

    if not df_desemprego.empty:
        df = filter_data_by_period(df_desemprego, period)
        fig = px.line(df, y=df.columns[0], title="Desemprego")
        st.plotly_chart(fig, use_container_width=True)

if not df_vix.empty:
    df = filter_data_by_period(df_vix, period)
    fig = px.line(df, y='Close', title="VIX (Volatilidade EUA)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Snapshots Fundamentalistas")
ticker = st.selectbox("Selecione um ticker", TICKERS)
data = load_fundamental_snapshot(ticker)

if data:
    st.json(data)
else:
    st.info("Snapshot fundamentalista não encontrado para este ticker.")
