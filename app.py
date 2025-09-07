import streamlit as st
from transformers import pipeline, AutoTokenizer
import torch
import re

# Modelo e tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 512
tokenizer.truncation_side = "right"

# Pipeline (usamos o .model por baixo para processar tensores por janela)
avaliador = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    truncation=True,
)

# Função para pré-processar texto
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zá-ú\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Inferência multi-janela com sobreposição e agregação
def predizer_multijanela(texto, max_length=512, stride=64, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_overflowing_tokens=True,
        stride=stride,
    )

    model = avaliador.model.to(device)
    model.eval()

    logits_list = []
    with torch.no_grad():
        total_chunks = enc["input_ids"].size(0)
        for i in range(total_chunks):
            batch = {
                "input_ids": enc["input_ids"][i:i+1].to(device),
                "attention_mask": enc["attention_mask"][i:i+1].to(device),
            }
            if "token_type_ids" in enc:
                batch["token_type_ids"] = enc["token_type_ids"][i:i+1].to(device)

            out = model(**batch)
            logits_list.append(out.logits.squeeze(0).cpu())

    # Agrega as janelas (média dos logits)
    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    probs = torch.softmax(logits, dim=-1)

    pred_idx = int(torch.argmax(probs).item())
    id2label = getattr(model.config, "id2label", None)
    if id2label and pred_idx in id2label:
        label = id2label[pred_idx]
    else:
        # Fallback: modelos desse repo usam labels "1 star", "2 stars", ...
        label = f"{pred_idx + 1} star"

    return {"label": label, "score": float(probs[pred_idx].item())}

# Função para avaliar redação e gerar nota e feedback
def avaliar_redacao(texto_redacao):
    texto_processado = preprocessar_texto(texto_redacao)
    resultado = predizer_multijanela(texto_processado, max_length=512, stride=64)
    # Label do modelo é tipo "1 star", "2 stars", etc. Pegamos o primeiro caractere numérico.
    try:
        estrelas = int(str(resultado["label"]).strip()[0])
    except Exception:
        estrelas = 3  # fallback neutro

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
