from flask import Flask, request, render_template_string
from transformers import pipeline
import re

app = Flask(__name__)

# Carregar pipeline de análise de sentimento (como proxy para avaliação)
# Em um sistema real, usaríamos um modelo treinado para avaliação de redações
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
    # O modelo nlptown retorna labels de 1 a 5 estrelas, vamos converter para nota 0-10
    estrelas = int(resultado['label'][0])
    nota = estrelas * 2  # escala 0-10
    feedback = {
        2: "Redação precisa melhorar a coerência e clareza.",
        4: "Boa redação, com ideias claras e bem estruturadas.",
        6: "Excelente redação, muito bem escrita e argumentada."
    }
    # Escolher feedback aproximado
    if nota <= 4:
        texto_feedback = feedback[2]
    elif nota <= 8:
        texto_feedback = feedback[4]
    else:
        texto_feedback = feedback[6]
    return nota, texto_feedback

# Página HTML simples para interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8" />
    <title>Avaliação Automática de Redações</title>
</head>
<body>
    <h1>Avaliação Automática de Redações</h1>
    <form method="post">
        <textarea name="redacao" rows="10" cols="80" placeholder="Digite sua redação aqui...">{{ redacao }}</textarea><br/>
        <button type="submit">Avaliar</button>
    </form>
    {% if nota is not none %}
        <h2>Nota atribuída: {{ nota }}/10</h2>
        <p><strong>Feedback:</strong> {{ feedback }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    nota = None
    feedback = None
    redacao = ""
    if request.method == "POST":
        redacao = request.form.get("redacao", "")
        if redacao.strip():
            nota, feedback = avaliar_redacao(redacao)
    return render_template_string(HTML_TEMPLATE, nota=nota, feedback=feedback, redacao=redacao)

if __name__ == "__main__":
    app.run(debug=True)
