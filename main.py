import streamlit as st
from transformers import BertTokenizerFast, TFBertForQuestionAnswering
import tensorflow as tf
import json

# Charger le modèle et le tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Charger le fichier JSON de données
with open("bio.json") as f:
    squad_data = json.load(f)

# Fonction pour obtenir la réponse prédite par le modèle
def get_predicted_answer(question, context):
    # Tokenizer en gardant une longueur maximale plus flexible
    inputs = tokenizer(question, context, return_tensors="tf", truncation=True, padding="max_length", max_length=512)
    
    # Prédiction des logits de début et de fin
    outputs = model(inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Trouver les positions des tokens de début et de fin avec les scores les plus élevés
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0] + 1

    # Vérification si la position de fin est après celle du début
    if end_idx < start_idx:
        end_idx = start_idx + 1

    # Convertir les tokens en texte
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Nettoyage pour éviter les erreurs dans la réponse
    if len(answer.strip()) == 0:
        answer = "Je ne peux pas trouver une réponse appropriée dans le contexte fourni."
    
    return answer

# Interface Streamlit
st.set_page_config(page_title="Medical Question-Answering Bot", page_icon="🤖")
st.title("Medical Question-Answering Bot")

# Message dans la barre latérale
with st.sidebar:
    st.title("Paramètres")
    st.write("Entrez le contexte et la question pour obtenir une réponse basée sur le modèle de QA.")

# Prendre les entrées de l'utilisateur
user_context = st.text_area("Entrez le texte ou le contexte ici...")
user_question = st.text_input("Tapez votre question ici...")

# Si le contexte et la question sont fournis, obtenir la réponse
if user_context and user_question:
    predicted_answer = get_predicted_answer(user_question, user_context)

    # Afficher la réponse prédite
    st.write(f"Réponse : {predicted_answer}")

