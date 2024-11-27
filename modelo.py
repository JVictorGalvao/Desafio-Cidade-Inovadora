import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
df = pd.read_csv('dados_simulados_onibus_30dias.csv')

# Converter variáveis categóricas em numéricas
df["Clima"] = df["Clima"].map({"Sol": 0, "Nublado": 1, "Chuva": 2})
df["Evento Local"] = df["Evento Local"].map({"Sim": 1, "Não": 0})

# One-hot encoding para a coluna 'Linha' (cada linha de ônibus vira uma coluna)
df = pd.get_dummies(df, columns=["Linha"])

# Selecionar as variáveis de entrada (X) e a variável alvo (y)
# Remover a coluna 'Demanda (%)' e 'Viagem' da entrada
X = df.drop(columns=["Demanda (%)", "Viagem"])
y = df["Demanda (%)"]  # A variável de saída é a 'Demanda (%)'

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Escalar os dados para o intervalo [0, 1]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print('SHAPE', X_train.shape)

# Construir o modelo de rede neural simples
modelo = Sequential([
    # Camada de entrada
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.3),  # Dropout para evitar overfitting
    Dense(64, activation='relu'),  # Camada oculta
    Dense(32, activation='relu'),  # Camada oculta adicional
    Dense(1, activation='linear')  # Camada de saída (previsão contínua)
])

# Compilar o modelo
modelo.compile(optimizer=Adam(learning_rate=0.001),
               loss='mean_squared_error', metrics=['mae'])

# Treinar o modelo
historico = modelo.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled),
                       epochs=20, batch_size=32, verbose=1)

# Avaliar o modelo
loss, mae = modelo.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Loss (MSE): {loss:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Fazer previsões
y_pred_scaled = modelo.predict(X_test_scaled)

# Desnormalizar as previsões
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Avaliar a performance do modelo
mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

modelo.save("modelo.keras")
