import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Carregar o modelo treinado
model = load_model("modelo_preditivo_neural_sem_escala.keras")

# Carregar os dados do CSV (substitua com o caminho correto para seu arquivo)
df = pd.read_csv('dados_simulados_onibus_30dias_com_lat_long.csv')

# Escalonar os dados de entrada
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Ajuste do scaler com base no CSV de treino
# Remover as colunas de lat/lon e target
X_train = df.drop(columns=["Demanda (%)", "Latitude", "Longitude"])
scaler_X.fit(X_train)
scaler_y.fit(df["Demanda (%)"].values.reshape(-1, 1))

# Função para gerar previsões com o modelo real


def prever_demanda(linha, parada, clima, evento_local):
    # Mapear os inputs para o modelo
    evento_map = {"Não": 0, "Sim": 1}
    linhas_map = {"1500": [1, 0, 0, 0, 0, 0],
                  "3200": [0, 1, 0, 0, 0, 0],
                  "5120": [0, 0, 1, 0, 0, 0],
                  "2300": [0, 0, 0, 1, 0, 0],
                  "1519": [0, 0, 0, 0, 1, 0],
                  "5100": [0, 0, 0, 0, 0, 1]}

    # Verificar se o clima e o evento são válidos
    if clima not in clima_map:
        raise ValueError(
            f"Clima inválido. Opções válidas: 'Sol', 'Nublado', 'Chuva'.")
    if evento_local not in evento_map:
        raise ValueError(
            f"Evento Local inválido. Opções válidas: 'Sim', 'Não'.")

    # Criar vetor de entrada para o modelo
    entrada = [15, parada, clima_map[clima],
               evento_map[evento_local], 105] + linhas_map[linha]

    # Escalonar os dados de entrada
    entrada_scaled = scaler_X.transform([entrada])

    # Prever a demanda
    previsao_scaled = model.predict(entrada_scaled)
    previsao = scaler_y.inverse_transform(previsao_scaled)

    return previsao[0][0]


# ---------------------------- INFORMAÇÕES SOBRE AS LINHAS (Esquerda) -----------------------------
st.title("Previsão de Demanda de Transporte Público")

# Entrada do usuário para previsões
linha = st.selectbox("Selecione a Linha", df['Linha'].unique())
parada = st.number_input(
    "Número da Parada", min_value=1, max_value=150, step=1)
evento_local = st.selectbox("Evento Local", ["Não", "Sim"])

# Exibir a previsão
if st.button("Gerar Previsão"):
    previsao = prever_demanda(linha, parada, clima, evento_local)
    st.write(f"A demanda prevista para a Linha {
             linha} na Parada {parada} é: {previsao:.2f}%")
