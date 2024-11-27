import streamlit as st
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium

st.set_page_config(
    page_title="streamlit-folium documentation",
    page_icon=":world_map:️",
    layout="wide",
)
# Carregar o modelo treinado
model = load_model("modelo_preditivo_neural_sem_escala.keras")

# Carregar dados do CSV (substitua com o caminho correto para seu arquivo)
df = pd.read_csv('dados_simulados_onibus_30dias.csv')
df_lat = pd.read_csv('pontos_lat_long.csv')

# Configuração do título
st.title("Dashboard de Previsão de Demanda de Ônibus")

# Organizar as colunas para o layout
col1, col2, col3 = st.columns([1, 2, 1])

# **Coluna 1 - Esquerda** (Informações das Linhas)
with col1:
    # Verificar as primeiras linhas do DataFrame
    st.header("Dados de Linhas de Ônibus")
    st.write(df.head())

    # Listar todas as linhas
    linhas = df['Linha'].unique()
    st.header("Linhas de Ônibus Disponíveis")
    st.write(linhas)

    # Função para gerar previsões aleatórias
    def gerar_previsao_aleatoria(linhas):
        previsoes = []
        for linha in linhas:
            previsao = random.uniform(10, 120)
            previsoes.append((linha, round(previsao, 2)))
        return previsoes

    # Exibir 10 previsões aleatórias
    st.header("10 Previsões Aleatórias de Demanda")
    previsoes = gerar_previsao_aleatoria(linhas)
    for linha, previsao in previsoes[:10]:  # Limitar a 10 previsões
        st.write(f"Linha: {linha} - Demanda Prevista: {previsao}%")

    # Se o usuário escolher uma linha, mostrar a previsão com o modelo
    linha_escolhida = st.selectbox('Escolha uma Linha de Ônibus:', linhas)

    if linha_escolhida:
        st.header(f"Previsão de Demanda para a Linha {linha_escolhida}")

        # Filtrar o DataFrame para a linha escolhida
        df_linha = df[df['Linha'] == linha_escolhida]

        # Selecionar as features de entrada para o modelo
        # Supondo que você tenha variáveis como 'Parada', 'Clima', 'Evento Local' etc.
        # Ajuste conforme necessário
        X = df_linha[['Parada', 'Clima', 'Evento Local']]
        X_scaled = MinMaxScaler().fit_transform(X)  # Normalização

        # Gerar as previsões
        previsao_demanda = model.predict(X_scaled)

        # Adicionar a previsão ao DataFrame
        df_linha['Previsão de Demanda (%)'] = previsao_demanda

        # Exibir as previsões
        st.write(df_linha[['Parada', 'Previsão de Demanda (%)']])

# **Coluna 2 - Centro** (Gráficos)
with col2:
    # Gráfico para visualizar as linhas com mais lotação
    st.header("Linhas com Mais Lotação")
    # Seleciona 5 linhas aleatórias
    linhas_aleatorias = random.sample(list(df['Linha'].unique()), 5)

# Calcular a média de demanda para essas linhas
    df_aleatorio = df[df['Linha'].isin(linhas_aleatorias)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Linha', y='Demanda (%)',
                data=df_aleatorio, palette='viridis')
    plt.title('Distribuição de Demanda por Linha (Aleatória)')
    plt.xlabel('Linha')
    plt.ylabel('Demanda (%)')
    st.pyplot(plt)
    # Gráfico para visualizar as paradas com maior demanda
    st.header("Demanda por Parada em Cada Linha")
    demanda_por_parada = df.groupby(['Linha', 'Parada'])[
        'Demanda (%)'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=demanda_por_parada, x='Parada',
                 y='Demanda (%)', hue='Linha', marker='o')
    plt.title('Demanda por Parada para Cada Linha')
    plt.xlabel('Número da Parada')
    plt.ylabel('Demanda (%)')
    plt.legend(title='Linhas')
    st.pyplot(plt)

# **Coluna 3 - Direita** (Mapa)
with col3:
    st.header("Mapa das Paradas de Ônibus")

    # Criando um mapa centrado em uma coordenada específica (exemplo: João Pessoa)
    mapa = folium.Map(location=[-7.115, -34.861], zoom_start=12)

    # Adicionar marcadores para as paradas
    marker_cluster = MarkerCluster().add_to(mapa)

    # Suponha que o DataFrame tenha as colunas 'Latitude' e 'Longitude' para as paradas
    for index, row in df_lat.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    # Renderizar o mapa
    st_data = st_folium(mapa, width=725)
