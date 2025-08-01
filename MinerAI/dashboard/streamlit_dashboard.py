# dashboard/streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configurações da página
st.set_page_config(
    page_title="MinerAI Dashboard",
    page_icon="🎮",
    layout="wide"
)

# Título
st.title("🎮 MinerAI Dashboard")
st.markdown("Monitoramento em tempo real do agente de IA para o jogo *Miner*.")

# Sidebar
st.sidebar.header("Controles")
refresh = st.sidebar.button("🔄 Atualizar Dados")
auto_refresh = st.sidebar.checkbox("Atualização automática", value=True)
refresh_interval = st.sidebar.slider("Intervalo (seg)", 1, 10, 5)

# Funções de carregamento de dados
def load_training_logs(log_dir="logs/tensorboard"):
    """Carrega logs do TensorBoard (simulando com CSV ou JSON)"""
    # Este é um exemplo simulado. Em produção, você pode usar `tensorboardX` ou `pandas` com eventos.
    fake_data = {
        "step": list(range(0, 100_000, 1000)),
        "reward": np.random.normal(2.5, 1.0, 100).cumsum() * 0.1,
        "loss": np.random.normal(0.5, 0.1, 100).cumsum() * -0.01 + 0.8,
        "entropy": np.random.normal(0.8, 0.1, 100)
    }
    return pd.DataFrame(fake_data)

def load_game_stats():
    """Carrega estatísticas do jogo (simuladas ou reais)"""
    stats = {
        "Métrica": [
            "Rodadas Totais",
            "Vitórias",
            "Derrotas",
            "Taxa de Vitória",
            "Multiplicador Médio",
            "ROI (%)",
            "Lucro Total (BTC)"
        ],
        "Valor": [
            1000,
            782,
            218,
            "78.2%",
            "3.45x",
            "+24.6%",
            "0.124"
        ]
    }
    return pd.DataFrame(stats)

def load_recent_games():
    """Últimas rodadas jogadas"""
    return pd.DataFrame({
        "Rodada": range(1, 11),
        "Tiles Clicados": np.random.randint(1, 10, 10),
        "Resultado": np.random.choice(["Vitória", "Derrota"], 10, p=[0.8, 0.2]),
        "Multiplicador": [f"{np.random.uniform(1.2, 5.0):.2f}x" for _ in range(10)],
        "Data": [datetime.now().strftime("%H:%M:%S")] * 10
    })

# Carregar dados
if auto_refresh or refresh:
    with st.spinner("Carregando dados..."):
        df_train = load_training_logs()
        df_stats = load_game_stats()
        df_recent = load_recent_games()

    st.success("Dados carregados com sucesso!")

# Abas
tab1, tab2, tab3 = st.tabs(["📈 Treinamento", "📊 Desempenho", "📋 Histórico"])

# === ABAS ===

# 📈 Aba 1: Treinamento
with tab1:
    st.subheader("Evolução do Treinamento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1 = px.line(df_train, x="step", y="reward", title="Recompensa por Passo")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(df_train, x="step", y="loss", title="Perda de Treinamento")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        fig3 = px.line(df_train, x="step", y="entropy", title="Entropia da Política")
        st.plotly_chart(fig3, use_container_width=True)

# 📊 Aba 2: Desempenho
with tab2:
    st.subheader("Estatísticas do Agente")
    
    # Tabela de métricas
    st.dataframe(df_stats, use_container_width=True)
    
    # Gráfico de pizza: Vitórias vs Derrotas
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Vitórias", "Derrotas"],
        values=[782, 218],
        marker_colors=["#4CAF50", "#F44336"]
    )])
    fig_pie.update_layout(title="Distribuição de Resultados")
    st.plotly_chart(fig_pie, use_container_width=True)

# 📋 Aba 3: Histórico
with tab3:
    st.subheader("Últimas 10 Rodadas")
    st.dataframe(df_recent, use_container_width=True)

    # Botão para simular nova rodada
    if st.button("▶️ Simular Nova Rodada"):
        st.info("Rodada simulada! Atualizando estatísticas...")
        # Aqui você poderia chamar `evaluate.py` ou `miner_bot.py`

# Rodapé
st.markdown("---")
st.markdown("🛠️ **MinerAI** – Agente de IA para jogos de risco | Desenvolvido com ❤️ para estudo e pesquisa")