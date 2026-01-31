#----------------------ETAPA 1: IMPORTAR MÓDULOS IMPORTANTES----------------------

import streamlit as st  # A biblioteca que transforma Python em Site Web
import joblib           # A ferramenta para carregar o "cérebro" da IA que salvamos antes
import numpy as np      # Ferramenta para organizar os dados numéricos

# 1. Configurando a aba do navegador
# page_title: O nome que aparece na aba lá em cima (como no Google ou Facebook)
# page_icon: O desenhinho (favicon) ao lado do nome
st.set_page_config(page_title="Portal do Consultor - Churn", page_icon="📈")

# 2. Textos da Tela Principal
st.title("🛡️ Sistema de Retenção de Clientes") # O título grande (H1)
st.markdown("Insira os dados do cliente para verificar o risco de cancelamento.") # Texto explicativo


#----------------------ETAPA 2: CARREGANDO A INTELIGÊNCIA (O CÉREBRO)----------------------
# Carregando os arquivos .pkl que geramos no outro script. O site NÃO está treinando a IA de novo. 
# Ele está apenas LENDO o que já foi aprendido. É instantâneo.
modelo = joblib.load('modelo_churn_v1.pkl')      # Carrega as regras de decisão (Random Forest)
scaler = joblib.load('padronizador_v1.pkl')      # Carrega a régua matemática (StandardScaler)


#----------------------ETAPA 3: CRIANDO A INTERFACE DE ENTRADA (FORMULÁRIO)----------------------
# Criando duas colunas para o visual ficar mais organizado (lado a lado)
col1, col2 = st.columns(2)

# Na coluna da esquerda (col1)
with col1:
    # Campo para digitar números.
    # min_value=1: Impede que alguém digite 0 ou número negativo.
    # value=12: Já deixa o número 12 preenchido como padrão.
    tempo = st.number_input("Tempo de Contrato (meses)", min_value=1, value=12)
    
    # Campo para valor financeiro.
    valor = st.number_input("Valor da Fatura (R$)", min_value=0.0, value=70.0)

# Na coluna da direita (col2)
with col2:
    # Slider: Aquela barrinha de arrastar. Ótimo para notas ou contagens pequenas.
    queixas = st.slider("Histórico de Reclamações", 0, 10, 1)


#----------------------ETAPA 4: PROCESSAMENTO DE DADOS----------------------
# O código dentro do 'if' só roda quando o botão é clicado
if st.button("🔍 Analisar Risco"):
    
    # --- PASSO CRUCIAL: O Tradutor (Scaler) ---
    # O usuário digitou "70 reais". A IA aprendeu com números normalizados (ex: 0.5). Precisamos usar o MESMO scaler do treino para traduzir o dado novo.
    # Os colchetes duplos [[ ]] são necessários porque a IA espera uma tabela, não um número solto.
    dados = scaler.transform([[tempo, valor, queixas]])
    
    # --- A Previsão de Probabilidade ---
    # predict_proba: Em vez de só responder "Sim" ou "Não", a IA diz a CERTEZA dela. Retorna algo como: [0.20, 0.80] -> (20% de ficar, 80% de sair).
    # Pegamos o [0][1] para ver a chance da classe 1 (Cancelamento).
    probabilidade = modelo.predict_proba(dados)[0][1]


#----------------------ETAPA 5: FEEDBACK DE NEGÓCIOS----------------------
# Cria uma linha divisória visual
    st.divider()
    
    # Lógica do Semáforo (Traffic Light System):
    
    # CASO VERMELHO (Risco > 70%)
    if probabilidade > 0.7:
        # st.error cria uma caixa VERMELHA automática
        st.error(f"**ALTO RISCO DE SAÍDA!** ({probabilidade*100:.1f}%)")
        st.info("💡 **Sugestão Comercial:** Oferecer desconto de fidelidade imediato.")
        
    # CASO AMARELO (Risco entre 30% e 70%)
    elif probabilidade > 0.3:
        # st.warning cria uma caixa AMARELA
        st.warning(f"**Risco Moderado** ({probabilidade*100:.1f}%)")
        st.info("💡 **Sugestão Comercial:** Realizar chamada de acompanhamento.")
        
    # CASO VERDE (Risco < 30%)
    else:
        # st.success cria uma caixa VERDE
        st.success(f"**Cliente Estável** ({probabilidade*100:.1f}% de risco)")

