import pandas as pd
import spacy
import joblib # Para salvar o modelo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Carregar dados
print("Carregando dataset...")
df = pd.read_csv("dataset_chamados.csv")

# 2. Pipeline de processamento simplificado pra performance
# vamos usar o Spacy dentro do fluxo da UI ou pré-processamento.
nlp = spacy.load("pt_core_news_sm")

def preprocessar_rapido(texto):
    # Lematização básica
    doc = nlp(texto)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_punct])

print("Processando textos (Isso pode levar alguns segundos)...")
df['texto_limpo'] = df['texto'].apply(preprocessar_rapido)

# 3. Dividir Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(df['texto_limpo'], df['categoria'], test_size=0.2)

# 4. Criar e Treinar Pipeline ML
model_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_pipeline.fit(X_train, y_train)

# 5. Avaliar
print("\nRelatório de Precisão:")
predicoes = model_pipeline.predict(X_test)
print(classification_report(y_test, predicoes))

# 6. Salvar o cérebro
joblib.dump(model_pipeline, "modelo_triagem.pkl")
print("✅ Modelo treinado e salvo como 'modelo_triagem.pkl'")