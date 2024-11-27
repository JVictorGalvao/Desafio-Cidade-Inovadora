import streamlit_folium
import streamlit as st
import numpy as np
import pandas as pd
import random
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="streamlit-folium documentation",
    page_icon=":world_map:️",
    layout="wide",
)

# Carregar dados e modelo
# Ajuste o caminho do seu arquivo CSV
df = pd.read_csv('dados_simulados_onibus_30dias.csv')
# Arquivo com latitudes e longitudes das paradas
df_lat = pd.read_csv('pontos_lat_long.csv')

df_demanda = pd.read_csv('dados_demanda.csv')
# Carregando seu modelo
# model = load_model("modelo_preditivo.h5")

# Layout do Streamlit
st.markdown("<h1 style='text-align: center; color: white;'>Sistema Preditivo de Tráfego e Demanda de Transporte</h1>",
            unsafe_allow_html=True)

# Dividindo a página em três colunas
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# **Coluna 1 - Esquerda** (Informações gerais sobre o sistema)
with col1:
    df = pd.read_csv('dados_simulados_onibus_30dias.csv')

    # Converter variáveis categóricas em numéricas
    df["Clima"] = df["Clima"].map({"Sol": 0, "Nublado": 1, "Chuva": 2})
    df["Evento Local"] = df["Evento Local"].map({"Sim": 1, "Não": 0})

    # Exibir título
    st.markdown("<h2 style='text-align: center; color: white;'>Informações Gerais</h2>",
                unsafe_allow_html=True)
    # Obter as linhas disponíveis
    linhas = df['Linha'].unique()

    # Criar uma linha horizontal para exibir as linhas de ônibus
    st.markdown("<h3 style='text-align: center;'>Linhas de Ônibus Disponíveis</h3>",
                unsafe_allow_html=True)
    cols = st.columns(len(linhas))  # Criar uma coluna para cada linha

    # Preencher cada coluna com o nome de uma linha
    for col, linha in zip(cols, linhas):
        # Ícone de ônibus para decorar (opcional)
        col.write(f"🚌 Linha {linha}")

    # Obter as linhas disponíveis
    linhas = df['Linha'].unique()

    # Adicionar dados fictícios para o número de ônibus em rota e a parada atual
    df['Onibus_ID'] = df.groupby(
        'Linha').cumcount() + 1  # ID fictício de ônibus
    df['Parada Atual'] = df['Parada'] % 10  # Paradas atuais simuladas

    # Criar uma lista suspensa para selecionar uma linha
    st.markdown("<h3 style='text-align: center;'>Selecione uma Linha de Ônibus</h3>",
                unsafe_allow_html=True)

    linha_selecionada = st.selectbox(
        "Escolha uma linha para visualizar os dados:",
        options=df['Linha'].unique()
    )

    if linha_selecionada:
        # Filtrar dados da linha selecionada
        df_linha = df[df['Linha'] == linha_selecionada]

        # Cálculos gerais
        # Número de ônibus em rota
        num_onibus = random.randint(1, 4)
        dados_onibus = []
        for i in range(num_onibus):
            dados_onibus.append({
                'Onibus_ID': i + 1,
                'Parada Atual': random.randint(1, df_linha['Parada'].nunique()),
                'Demanda (%)': round(random.uniform(10, 120), 2)
            })
        num_paradas = df_linha['Parada'].nunique()  # Número total de paradas
        lotacao_media = sum(onibus['Demanda (%)']
                            # Lotação média
                            for onibus in dados_onibus) / len(dados_onibus)

        # Exibir informações gerais
        st.markdown("<h3 style='text-align: center;'>Informações da Linha</h3>",
                    unsafe_allow_html=True)
        st.write(f"#### Número de Ônibus em Rota: {num_onibus}")
        st.write(f"#### Número de Paradas: {num_paradas}")
        st.write(f"#### Lotação Média: {lotacao_media:.2f}%")
        st.write("")  # Adiciona uma linha em branco
        st.write("")  # Adiciona mais espaço
        # Informações detalhadas de cada ônibus
        st.markdown("<h3 style='text-align: center;'>Detalhes dos Ônibus em Rota</h3>",
                    unsafe_allow_html=True)
        for onibus in dados_onibus:
            st.write(
                f" ##### ID: {onibus['Onibus_ID']} | Parada Atual: {onibus['Parada Atual']} | Demanda (%): {onibus['Demanda (%)']}")


# **Coluna 2 - Centro** (Gráficos)
with col2:
    st.markdown("<h2 style='text-align: center; color: white;'>Gráficos</h2>",
                unsafe_allow_html=True)

    dados_linhas = {
        "Linha": ["1500", "3200", "5120", "2300", "1519", "5100"],
        "Demanda Média (%)": [75, 23, 85, 112, 50, 80]
    }

    df_manual = pd.DataFrame(dados_linhas)

    # Gráfico de barras - Linhas com maior lotação
    st.markdown("<h3 style='text-align: center; color: white;'>Lotação por linhas</h3>",
                unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_manual["Linha"],
                y=df_manual["Demanda Média (%)"], palette='viridis')
    plt.title('Linhas com Maior Lotação')
    plt.xlabel('Linha')
    plt.ylabel('Demanda Média (%)')
    st.pyplot(plt)

    # Gráfico de linhas - Demanda por Parada
    st.markdown("<h3 style='text-align: center; color: white;'>Demanda por parada</h3>",
                unsafe_allow_html=True)
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
    st.markdown("<h2 style='text-align: center; color: white;'>Mapa das Paradas</h2>",
                unsafe_allow_html=True)
    # st.header("Mapa das Paradas de Ônibus")

    import requests

    # Sua chave da API
    API_KEY = "AIzaSyCxcmyn24T1B_YCjhGUOfFqvoK2F1aVlOI"

    # Endereço da API Directions
    # url = "https://maps.googleapis.com/maps/api/directions/json"

    # # Parâmetros da solicitação
    # params = {
    #     "origin": "Endereço de origem ou coordenadas",  # Ponto inicial
    #     "destination": "Endereço de destino ou coordenadas",  # Ponto final
    #     "departure_time": "now",  # Tempo atual para trânsito em tempo real
    #     "traffic_model": "best_guess",  # Modelo de trânsito
    #     "key": API_KEY
    # }

    # # Fazer a solicitação à API
    # response = requests.get(url, params=params)
    # data = response.json()

    # # Exibir informações úteis
    # if response.status_code == 200:
    #     print("Duração com trânsito:", data['routes']
    #           [0]['legs'][0]['duration_in_traffic']['text'])
    # else:
    #     print("Erro:", data['error_message'])

    url_places = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # Coordenadas centrais e tipo de lugar
    params_places = {
        "location": "-7.115, -34.861",  # Exemplo: João Pessoa
        "radius": "20000",  # Raio em metros
        "type": "bus_stop, bus_station",  # Tipo de lugar
        "key": API_KEY
    }

    response_places = requests.get(url_places, params=params_places)
    data_places = response_places.json()

    # st.write(data_places)
    st.markdown("<h3 style='text-align: center;'>Paradas de Ônibus Próximas</h3>",
                unsafe_allow_html=True)
    # Exibir paradas de ônibus próximas
    if response_places.status_code == 200:
        for place in data_places['results']:
            print("Nome:", place['name'])
            print("Endereço:", place.get('vicinity', 'Desconhecido'))
            print("Coordenadas:", place['geometry']['location'])
    else:
        print("Erro:", data_places['error_message'])

    mapa = folium.Map(location=[-7.115, -34.861], zoom_start=14)

# Adicionar marcadores para as paradas
    for place in data_places['results']:
        lat = place['geometry']['location']['lat']
        lng = place['geometry']['location']['lng']
        name = place['name']
        folium.Marker(location=[lat, lng], popup=name, icon=folium.Icon(
            icon="bus", prefix="fa", color="blue")).add_to(mapa)

    # Exibir mapa no Streamlit
    st_folium(mapa, width=800, height=600)

    # mapa = folium.Map(location=[-7.115, -34.861], zoom_start=12)
    # marker_cluster = MarkerCluster().add_to(mapa)

    # # Adicionar marcadores para as paradas no mapa
    # for index, row in df_lat.iterrows():
    #     folium.Marker(
    #         location=[row['Latitude'], row['Longitude']],
    #         icon=folium.Icon(color='blue')
    #     ).add_to(marker_cluster)

    # # Exibir o mapa
    # st_folium(mapa, width=725)

with col4:
    st.markdown("<h2 style='text-align: center; color: white;'>Previsão de ocupação nas próximas viagens</h2>",
                unsafe_allow_html=True)

    # Função para calcular a ocupação com base nas condições

    def simular_ocupacao(parada, total_paradas, linha, clima, evento_local, previsao_transito):
        # Normalizar a parada para estar entre 0 e 1
        x = parada / total_paradas

        # Gerar a demanda base com variação para cada linha
        if linha == '1500':  # Linha 1500: alta no início, baixa no fim
            demanda = 80 * np.sin(np.pi * x)  # Aumento e diminuição suave
        elif linha == '5100':  # Linha 5100: média no início, alta no meio, baixa no fim
            demanda = 50 + 30 * np.cos(np.pi * x)  # Pico no meio
        elif linha == '5120':  # Linha 5120: média no início, baixa no meio, alta no fim
            demanda = 50 + 30 * np.sin(np.pi * x)  # Pico no fim
        elif linha == '1519':  # Linha 1519: baixa no início, alta no meio, baixa no fim
            demanda = 20 + 50 * np.cos(np.pi * x)  # Pico no meio
        elif linha == '2300':  # Linha 2300: estável com leve aumento no início
            demanda = 60 + 20 * np.sin(np.pi * x / 2)  # Leve pico no início
        elif linha == '3200':  # Linha 3200: estável com leve aumento no fim
            demanda = 60 + 20 * np.sin(np.pi * x)  # Leve pico no fim
        else:
            demanda = 50  # Caso padrão, demanda constante

        # Modificar a demanda com base nas condições
        if clima == 'Sol':
            # Menor demanda em dias ensolarados, variação maior
            demanda -= np.random.uniform(15, 25)
        elif clima == 'Chuva':
            # Maior demanda em dias chuvosos, variação maior
            demanda += np.random.uniform(20, 40)
        elif clima == 'Nublado':
            # Demanda média em dias nublados, variação maior
            demanda += np.random.uniform(10, 30)

        # Aumentar ou diminuir a demanda em eventos locais
        if evento_local == 'Sim':
            # 85% de chance de aumentar a demanda
            evento_efeito = np.random.choice([1, -1], p=[0.85, 0.15])
            # Aumento ou diminuição de demanda
            demanda += np.random.uniform(30, 60) * evento_efeito

        # Ajustar a demanda com base na previsão de trânsito
        if previsao_transito == 'Pesado':
            # Aumento da demanda em trânsito pesado
            demanda += np.random.uniform(15, 35)
        elif previsao_transito == 'Leve':
            # Redução da demanda em trânsito leve
            demanda -= np.random.uniform(5, 15)

        # Garantir que a demanda nunca seja negativa e que não ultrapasse 120%
        demanda = np.clip(demanda, 10, 120)

        return demanda

    # Interface do Streamlit
    st.markdown("<h3 style='text-align: center;'>Simulação de Ocupação por Linha</h3>",
                unsafe_allow_html=True)

    # Seleção da linha de ônibus
    linha_selecionada = st.selectbox(
        'Escolha uma Linha de Ônibus:', df['Linha'].unique())

    # Condições de simulação
    clima_selecionado = st.selectbox(
        'Escolha o clima:', ['Sol', 'Chuva', 'Nublado'])
    evento_selecionado = st.selectbox('Há eventos locais?', ['Sim', 'Não'])
    transito_selecionado = st.selectbox(
        'Qual a previsão de trânsito?', ['Leve', 'Pesado'])

    # Botão para gerar a simulação
    if st.button('Gerar Simulação'):
        if linha_selecionada and clima_selecionado and evento_selecionado and transito_selecionado:
            # Gerar a ocupação simulada conforme as condições selecionadas
            total_paradas = df[df['Linha'] ==
                               linha_selecionada]['Parada'].nunique()
            ocupacao_simulada = []

            # Gerar demanda simulada para cada parada
            for parada in range(1, total_paradas + 1):
                demanda = simular_ocupacao(parada, total_paradas, linha_selecionada,
                                           clima_selecionado, evento_selecionado, transito_selecionado)
                ocupacao_simulada.append(demanda)

            # Calcular a média da ocupação
            media_ocupacao = np.mean(ocupacao_simulada)

            # Mostrar os resultados da simulação
            st.write(f"Simulação de Ocupação para a Linha {
                linha_selecionada} com as condições selecionadas:")

            # Exibir a média de ocupação
            st.write(f"#### A média de ocupação para a linha {
                linha_selecionada} é: {round(media_ocupacao, 2)}%")
            # Exibir gráfico ou tabela com os resultados
            result_df = pd.DataFrame({
                'Parada': range(1, len(ocupacao_simulada) + 1),
                'Ocupação Simulada (%)': ocupacao_simulada
            })
            # st.write(result_df)

            # Exibir gráfico
            st.line_chart(result_df.set_index('Parada'))
        else:
            st.warning(
                'Por favor, selecione todos os parâmetros para a simulação.')
