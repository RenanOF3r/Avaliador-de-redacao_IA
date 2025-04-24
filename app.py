import streamlit as st
from transformers import pipeline
import re

# Carregar pipeline de análise de sentimento (proxy para avaliação)
avaliador = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Função para pré-processar texto
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zá-ú\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Função para avaliar redação e gerar nota e feedback
def avaliar_redacao(texto_redacao):
    texto_processado = preprocessar_texto(texto_redacao)
    resultado = avaliador(texto_processado)[0]
    estrelas = int(resultado['label'][0])
    nota = estrelas * 2  # escala 0-10
    feedback = {
        2: "Redação precisa melhorar a coerência e clareza.",
        4: "Boa redação, com ideias claras e bem estruturadas.",
        6: "Excelente redação, muito bem escrita e argumentada."
    }
    if nota <= 4:
        texto_feedback = feedback[2]
    elif nota <= 8:
        texto_feedback = feedback[4]
    else:
        texto_feedback = feedback[6]
    return nota, texto_feedback

# Interface Streamlit
st.title("Avaliação Automática de Redações com IA")

redacao = st.text_area("Digite sua redação aqui:", height=200)

if st.button("Avaliar"):
    if redacao.strip():
        nota, feedback = avaliar_redacao(redacao)
        st.success(f"Nota atribuída: {nota}/10")
        st.info(f"Feedback: {feedback}")
    else:
        st.warning("Por favor, digite uma redação para avaliação.")
