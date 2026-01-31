import pandas as pd
import numpy as np

# Definindo a semente para que os resultados sejam reproduzíveis
np.random.seed(42)

# Gerando 500 registros
n_registros = 500

data = {
    'tempo_contrato': np.random.randint(1, 48, n_registros),  # 1 a 48 meses
    'valor_mensal': np.random.uniform(50.0, 150.0, n_registros).round(2), # R$ 50 a 150
    'reclamacoes': np.random.poisson(1.5, n_registros) # Média de 1.5 reclamações
}

df = pd.DataFrame(data)

# Criando uma lógica de Churn: 
# O cliente tem mais chance de sair se tiver muitas reclamações OU contrato curto
# (Isso ajuda a IA a aprender um padrão real)
df['cancelou'] = ((df['reclamacoes'] > 2) | (df['tempo_contrato'] < 6)).astype(int)

# Salvando em CSV de forma profissional (sem o índice do pandas)
df.to_csv('churn_data.csv', index=False)
print("Arquivo 'churn_data.csv' gerado com sucesso!")