import pandas as pd
import numpy as np
import random

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

# Lista para armazenar os dados
dados = []

# Função para gerar a demanda com base nos padrões de cada linha


def gerar_demanda(parada, total_paradas, linha):
    # Normalizar a parada para estar entre 0 e 1
    x = parada / total_paradas

    # Padrão para linha 1500: Alta no início, baixa no fim
    if linha == '1500':
        # Sin para variação de alta para baixa
        demanda = 80 * np.sin(np.pi * x)
    # Linha 5100: Média no início, alta no meio, baixa no fim
    elif linha == '5100':
        demanda = 50 + 30 * np.cos(np.pi * x)  # Pico no meio
    # Linha 5120: Média no início, baixa no meio, alta no fim
    elif linha == '5120':
        demanda = 50 + 30 * np.sin(np.pi * x)  # Pico no fim
    # Linha 1519: Baixa no início, alta no meio, baixa no fim
    elif linha == '1519':
        demanda = 20 + 60 * np.cos(np.pi * x)  # Pico no meio
    # Linha 2300: Estável com leve aumento no início
    elif linha == '2300':
        demanda = 60 + 20 * np.sin(np.pi * x / 2)  # Leve pico no início
    # Linha 3200: Estável com leve aumento no fim
    elif linha == '3200':
        demanda = 60 + 20 * np.sin(np.pi * x)  # Leve pico no fim
    else:
        demanda = 50  # Caso padrão, demanda constante

    # Garantir que a demanda nunca seja negativa e limitá-la para o máximo de 120
    demanda = max(demanda, 10)  # Evita valores negativos, mínimo 10%
    return min(demanda, 120)  # Limita a demanda máxima em 120%


# Gerar dados para 30 dias, 30 viagens por linha por dia
for dia in range(1, 2):  # Para testar, gerando 1 dia (alterar para 30 para mais dias)
    for linha, info in linhas.items():
        total_paradas = info["paradas"]
        duracao = info["duracao"]

        for viagem in range(1, 2):  # 30 viagens por dia (alterar para 30 para mais viagens)
            for parada in range(1, total_paradas + 1):
                demanda = gerar_demanda(parada, total_paradas, linha)
                dados.append({
                    "Linha": linha,
                    "Parada": parada,
                    "Demanda (%)": round(demanda, 2),
                })

# Converter para DataFrame
df = pd.DataFrame(dados)

# Salvar como CSV
# Exibir os primeiros registros
print(df.head())

# Salvar como CSV
df.to_csv("dados_demanda.csv", index=False)

# Exibir os primeiros registros
print(df)
