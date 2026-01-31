#EXERCÍCIO 1 - treinando IA com dados prontos (sem uso de .csv)

#-------------ETAPA 1: IMPORTAR TODOS OS MÓDULOS NECESSÁRIOS-------------

import pandas as pd  # Ferramenta para criar e mexer em tabelas (Excel do Python)
import numpy as np   # Ferramenta para matemática pesada e arrays numéricos

# Do Scikit-Learn (a caixa de ferramentas de IA), pegamos peças específicas:
from sklearn.model_selection import train_test_split # Para dividir a "prova" do "estudo"
from sklearn.preprocessing import StandardScaler      # Para colocar todos os números na mesma régua
from sklearn.ensemble import RandomForestClassifier  # O cérebro da IA (Floresta de Árvores)
from sklearn.metrics import classification_report, confusion_matrix # Para corrigir a prova e dar a nota

import seaborn as sns           # Ferramenta para gráficos bonitos
import matplotlib.pyplot as plt # Ferramenta base para desenhar gráficos
import joblib                   # Ferramenta para salvar o trabalho (o "Save Game")


# #-------------ETAPA 2: CRIAR UM DICIONÁRIO COM DADOS BRUTOS PARA SEREM ANALISADOS-------------
# # Criando um dicionário (dados brutos)
# data = {
#     'tempo_contrato': [1,24,12,48,2,36,6,12,3,24], # Meses que o cliente está na empresa
#     'valor_mensal': [70,50,60,45,80,55,75,65,90,50], # Quanto ele paga
#     'reclamacoes': [3,0,1,0,4,1,2,0,5,0],            # Quantas vezes ligou reclamando
#     'cancelou': [1,0,0,0,1,0,1,0,1,0]                # O GABARITO: 1 = Cancelou, 0 = Ficou
# }

# # Transformando em Tabela (DataFrame) para o Pandas entender
# df = pd.DataFrame(data)



#-------------ETAPA 2.1: CRIAR UM DICIONÁRIO COM DADOS BRUTOS PARA SEREM ANALISADOS (SOMENTE DEPOIS QUE FINALIZAR A PRIMEIRA ETAPA)-------------
# Antes: Digitávamos um dicionário pequeno na mão.
# Agora: Lemos um arquivo CSV com 500 clientes (gerado pelo script anterior).

try:
    print("Carregando arquivo 'churn_data.csv'...")
    df = pd.read_csv('churn_data.csv') # O comando que lê o arquivo e cria a tabela
    print(f"Sucesso! {len(df)} linhas importadas.")
    
except FileNotFoundError:
    print("ERRO: O arquivo 'churn_data.csv' não foi encontrado na pasta!")
    print("Dica: Rode o script gerador de dados primeiro.")
    exit() # Para o programa se não tiver arquivo




#-------------ETAPA 3: REALIZAR O PRÉ-PROCESSAMENTO DE DADOS (PREPARANDO O TERRENO PARA O TREINAMENTO DA MÁQUINA)-------------
# 1. Separando as Perguntas (X) da Resposta (y)
# X = Tudo menos a coluna 'cancelou'. São as pistas que a IA vai olhar.
X = df.drop('cancelou', axis=1) 
# y = Apenas a coluna 'cancelou'. É o que queremos que ela aprenda a prever.
y = df['cancelou']

# 2. Dividindo o Simulado (Treino) da Prova Real (Teste)
# test_size=0.2: Guarda 20% dos dados num cofre para testar no final.
# random_state=42: Garante que o embaralhamento seja sempre o mesmo (para a aula dar certo para todos).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalizando (Colocando na mesma escala)
scaler = StandardScaler()

# ATENÇÃO ALUNOS:
# fit_transform no TREINO: A IA calcula a média e desvio padrão dos dados de estudo e já aplica.
X_train_scaled = scaler.fit_transform(X_train)

# transform no TESTE: Usamos a régua calculada no treino para medir o teste.
# NÃO fazemos "fit" no teste, pois seria "colar" (olhar dados do futuro).
X_test_scaled = scaler.transform(X_test)

print("\nDados normalizados e divididos!")




#-------------ETAPA 4: TREINAR O MODELO E REALIZAR PREVISÃO DE DADOS-------------
# Criando o Modelo
# n_estimators=100: Dizemos para criar 100 "árvores de decisão" que vão votar no resultado final.
modelo_churn = RandomForestClassifier(n_estimators=100, random_state=42)

# O comando FIT (Treinar/Ajustar)
# É aqui que a IA lê as perguntas (X_train) e as respostas (y_train) e aprende os padrões.
modelo_churn.fit(X_train_scaled, y_train)

# O comando PREDICT (Prever)
# A IA faz a "prova". Ela recebe as perguntas do teste (X_test) mas SEM as respostas.
previsoes = modelo_churn.predict(X_test_scaled)



#-------------ETAPA 5: AVALIAR O MODELO (É AQUI QUE SABEMOS SE A IA FOI BEM)-------------
print("\n--- Relatório de Performance ---")
# Compara o Gabarito Real (y_test) com o que a IA chutou (previsoes)
# Mostra Acurácia, Precisão e Recall.
print(classification_report(y_test, previsoes))


# Gerando a matriz matemática (quantos acertos vs erros)
cm = confusion_matrix(y_test, previsoes)

# Desenhando o gráfico colorido
plt.figure(figsize=(6,4)) # Define o tamanho da imagem
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # annot=True escreve os números nos quadrados
plt.xlabel('Previsão da IA')        # Eixo X
plt.ylabel('Valor Real (Gabarito)') # Eixo Y
plt.title('Matriz de Confusão')     # Título
plt.show() # Mostra a janela com o gráfico



#-------------ETAPA 6: DEPLOY (SALVANDO O TRABALHO)-------------
# Salvando o "Cérebro" treinado num arquivo
joblib.dump(modelo_churn, 'modelo_churn_v1.pkl')

# Salvando a "Régua" (Scaler).
# Dica pro aluno: Sem a régua, a IA não entende os números novos no futuro!
joblib.dump(scaler, 'padronizador_v1.pkl')

print("Arquivos de inteligência exportados com sucesso!")