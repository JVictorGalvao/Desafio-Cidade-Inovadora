from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization


# Configurações gerais
random.seed(42)

# Dados fornecidos
linhas = {
    "1500": {"paradas": 134, "duracao": 105},
    "3200": {"paradas": 88, "duracao": 89},
    "5120": {"paradas": 114, "duracao": 90},
    "2300": {"paradas": 90, "duracao": 60},
    "1519": {"paradas": 107, "duracao": 90},
    "5100": {"paradas": 148, "duracao": 111},
}

climas = ["Sol", "Chuva", "Nublado"]
eventos_locais = ["Sim", "Não"]

# Lista para armazenar os dados
dados = []

# Função para gerar a demanda orgânica


def gerar_demanda(parada, total_paradas):
    meio = total_paradas // 2
    if parada <= meio:
        return random.randint(10, 80) + int((parada / meio) * 40)
    else:
        return random.randint(10, 80) + int(((total_paradas - parada) / meio) * 40)


# Gerar dados para 30 dias, 30 viagens por linha por dia
for dia in range(1, 31):  # Dias de 1 a 30
    for linha, info in linhas.items():
        total_paradas = info["paradas"]
        duracao = info["duracao"]

        for viagem in range(1, 31):  # 30 viagens por dia
            clima = random.choice(climas)
            evento = random.choice(eventos_locais)

            for parada in range(1, total_paradas + 1):
                demanda = gerar_demanda(parada, total_paradas)
                # Limitar a demanda a no máximo 120%
                demanda = min(demanda, 120)
                dados.append({
                    "Dia": dia,
                    "Linha": linha,
                    "Viagem": viagem,
                    "Parada": parada,
                    "Clima": clima,
                    "Evento Local": evento,
                    "Demanda (%)": demanda,
                    "Duração Média (minutos)": duracao,
                })

# Converter para DataFrame
df = pd.DataFrame(dados)

# Salvar como CSV
df.to_csv("dados_simulados_onibus_30dias.csv", index=False)

# Exibir os primeiros registros
print(df.head())


# Carregar os dados
df = pd.read_csv("dados_simulados_onibus_30dias.csv")

# Converter variáveis categóricas em numéricas
df["Evento Local"] = df["Evento Local"].map({"Sim": 1, "Não": 0})
df["Clima"] = df["Clima"].map({"Sol": 0, "Nublado": 1, "Chuva": 2})
df = pd.get_dummies(df, columns=["Linha"])  # One-hot encoding para as linhas

# Separar recursos (X) e alvo (y)
X = df.drop(columns=["Demanda (%)", "Viagem"])
y = df["Demanda (%)"]

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Escalar os dados para o intervalo [0, 1]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Construir o modelo com camadas densas e regularização L2
model = Sequential([
    Dense(128, activation='relu', input_shape=(
        X_train_scaled.shape[1],), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),

    # Garante que os valores da previsão sejam >= 0
    Dense(1, activation='relu')
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error', metrics=['mae'])

# Resumo do modelo
model.summary()

# Treinar o modelo
historico = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=10,  # Ajuste o número de épocas conforme necessário
    batch_size=32,  # Ajuste o tamanho do lote
    verbose=1
)

# Avaliar o modelo
loss, mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Loss (MSE): {loss:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Fazer previsão
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Garantir que as previsões não sejam negativas
y_pred = np.clip(y_pred, 0, None)  # Força as previsões a serem >= 0

# Avaliar a performance do modelo
mae_final = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE) Final: {mae_final:.2f}")

model.save("modelo_preditivo_neural_sem_escala.keras")
