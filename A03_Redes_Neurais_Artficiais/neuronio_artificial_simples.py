import tensorflow as tf
import numpy as np

# 1. CRIANDO DADOS PARA TREINO (O "Simulador" de direção)
# Vamos criar situações hipotéticas para ensinar o carro.
# Entradas: [Distância (m), Velocidade (km/h)]
dados_treino = np.array([
    [100, 30],  # Longe e devagar -> Não frear (0)
    [10, 100],  # Perto e rápido -> FREAR AGORA! (1)
    [5, 10],    # Muito perto, mesmo devagar -> Frear (1)
    [50, 80],   # Distância média, rápido -> Perigo/Frear (1)
    [80, 40],   # Longe, velocidade média -> Não frear (0)
    [2, 2],     # Colado, quase parado -> Frear (1)
    [200, 120], # Muito longe, muito rápido -> Não frear (0)
], dtype=float)

# Saídas esperadas (Gabarito): 0 = Seguir, 1 = Frear
respostas_treino = np.array([0, 1, 1, 1, 0, 1, 0], dtype=float)

# 2. NORMALIZAÇÃO (Dica de Ouro para TensorFlow)
# Redes neurais odeiam números grandes (tipo 100, 200). Elas gostam de 0 a 1.
# Vamos dividir a distância por 200 (máx) e velocidade por 200 (máx estimado)
dados_treino_norm = dados_treino / 200.0

# 3. CRIANDO O CÉREBRO (Modelo Keras)
model = tf.keras.Sequential([
    # Camada de Entrada: Espera 2 números (Distância e Velocidade)
    # Camada Oculta: 4 neurônios para processar a relação entre velocidade/distância
    tf.keras.layers.Dense(4, input_shape=(2,), activation='relu'),
    
    # Camada de Saída: 1 neurônio (Decisão Final: 0 ou 1)
    # 'sigmoid' é perfeito para probabilidade (retorna entre 0 e 1)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 4. COMPILANDO (Configurando o aprendizado)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. TREINANDO
print("🚗 Iniciando treinamento de direção...")
# Epochs = Quantas vezes ele repete o treino. Como temos poucos dados, repetimos muitas vezes.
model.fit(dados_treino_norm, respostas_treino, epochs=500, verbose=0) 
print("Treinamento concluído!")

# 6. TESTE NO MUNDO REAL
def testar_freio(distancia, velocidade):
    # Precisamos normalizar os dados do teste igual fizemos no treino!
    teste = np.array([[distancia, velocidade]]) / 200.0
    
    # A IA prevê (retorna um número entre 0 e 1)
    probabilidade = model.predict(teste, verbose=0)[0][0]
    
    print(f"\nSituação: Distância {distancia}m | Velocidade {velocidade}km/h")
    print(f"Probabilidade de Batida: {probabilidade:.4f}")
    
    if probabilidade > 0.5:
        print("DECISÃO: 🛑 FREAR BRUSCAMENTE! 🛑")
    else:
        print("DECISÃO: 🟢 Seguir viagem.")

# Testando situações novas que a IA nunca viu
testar_freio(distancia=15, velocidade=90)  # Perto e Rápido (Perigo!)
testar_freio(distancia=150, velocidade=60) # Longe e Tranquilo