import streamlit as st
import joblib
import spacy
import pandas as pd

# Configuração da Página
st.set_page_config(page_title="Triagem de suporte", page_icon="🤖")

# --- Carregamento de recursos cacheado para não recarregar a cada clique ---
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_triagem.pkl")

@st.cache_resource
def carregar_nlp():
    return spacy.load("pt_core_news_sm")

try:
    modelo = carregar_modelo()
    nlp = carregar_nlp()
except:
    st.error("Erro: Execute primeiro o script 'treinar_modelo.py' para gerar o arquivo .pkl")
    st.stop()

# --- Lógica de Processamento ---
def analisar_chamado(texto_usuario):
    # 1. Processamento linguistico (Spacy)
    doc = nlp(texto_usuario)
    
    # Extrair entidades para mostrar inteligência
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Limpeza para o modelo ML
    texto_limpo = " ".join([token.lemma_.lower() for token in doc if not token.is_punct])
    
    # 2. Predição (Machine Learning)
    categoria_predita = modelo.predict([texto_limpo])[0]
    probs = modelo.predict_proba([texto_limpo])[0]
    confianca = max(probs) * 100
    
    return categoria_predita, confianca, entidades

# --- Interface Gráfica (Chat) ---
st.title("Triagem de suporte")
st.markdown("Descreva o problema em poucas palavras.")

# Inicializar histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Ex: O servidor AWS parou de responder..."):
    # 1. Exibir mensagem do usuário
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Processar a resposta da IA
    categoria, confianca, ents = analisar_chamado(prompt)
    
    # Montar resposta formatada
    resposta_md = f"""
    **Análise do Chamado:**
    - 📂 **Categoria:** `{categoria}`
    - 🎯 **Confiança:** `{confianca:.2f}%`
    """
    
    if ents:
        resposta_md += "\n\n**Entidades Detectadas:**"
        for ent in ents:
            resposta_md += f"\n- *{ent[0]}* ({ent[1]})"
    
    # Ação sugerida baseada na categoria
    acoes = {
        "Infraestrutura": "Encaminhando para equipe N2 - SysAdmin.",
        "Acesso": "Verificando logs de autenticação e AD.",
        "Hardware": "Abrindo ordem de serviço para suporte de campo.",
        "Software": "Verificando disponibilidade de licenças."
    }
    resposta_md += f"\n\n⚙️ **Ação:** {acoes.get(categoria, 'Triagem manual necessária.')}"

    # 3. Exibir resposta
    with st.chat_message("assistant"):
        st.markdown(resposta_md)
    st.session_state.messages.append({"role": "assistant", "content": resposta_md})