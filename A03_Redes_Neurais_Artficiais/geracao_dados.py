import pandas as pd
import random

# Vamos simular 2.000 situações registradas no passado
dados = []

print("💾 Gerando arquivo de histórico de direção...")

for _ in range(2000):
    distancia = random.randint(1, 200) # Metros
    velocidade = random.randint(0, 140) # Km/h
    
    # Esta é a lógica REAL do mundo físico (A física não muda)
    # Mas a IA NÃO VAI VER ESSE CÓDIGO. Ela só vai ver o resultado no Excel.
    deve_frear = 0 # (0 = Não)
    
    # Regras de segurança (Gabarito)
    if distancia < 30 and velocidade > 20:
        deve_frear = 1 # Perigo iminente
    elif distancia < 60 and velocidade > 60:
        deve_frear = 1 # Perigo médio
    elif distancia < 100 and velocidade > 100:
        deve_frear = 1 # Alta velocidade
        
    dados.append([distancia, velocidade, deve_frear])

# Salva no arquivo
df = pd.DataFrame(dados, columns=['distancia', 'velocidade', 'resultado_freio'])
df.to_csv('historico_piloto.csv', index=False)
print("✅ Arquivo 'historico_piloto.csv' criado com sucesso!")