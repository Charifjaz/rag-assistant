# frontend/streamlit_app.py
import sys
import os

# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.utils_streamlit import display_model_config
from app.rag_temp_engine import run_rag_on_uploaded_docs
from langchain_community.document_loaders import PyPDFLoader
import tempfile


# ✅ Les imports de ton application doivent venir APRÈS
import streamlit as st
from app.rag_engine import ask_question
from app import config
from app.utils import load_api_key  # ✅ Nouvelle version qui gère clé API utilisateur
import traceback


# 🎨 Configuration de la page
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🌟 Titre principal
st.title("🔎 RAG Chatbot - Question / Réponse augmentée")
st.markdown("""
Pose ta question ci-dessous ✉️
Le modèle te répondra à partir des documents PDF que tu as indexés.
""")

# 🧠 Barre latérale : Clé API et paramètres
with st.sidebar:
    # ⚙️ Paramètres du modèle (directement visibles)
    model, temperature, k = display_model_config("global")

     # 🔑 Clé OpenAI personnalisée
    user_api_key = st.text_input(
        "🔑 Ta clé OpenAI (optionnelle)",
        type="password",
        placeholder="sk-...",
        # help="Elle sera utilisée à la place de celle du .env si renseignée.",
    )


# 🖋️ Entrée utilisateur : Question
question = st.text_input(
    "❓ Ta question :",
    placeholder="Ex: Ask me ! The World is yours."
)

# Visualisation utile pour debug
# if "OPENAI_API_KEY_USED" in st.session_state:
#     source = st.session_state["OPENAI_API_SOURCE"]
#     masked_key = st.session_state["OPENAI_API_KEY_USED"][:6] + "..." + st.session_state["OPENAI_API_KEY_USED"][-4:]
#     st.info(f"🔐 Clé utilisée ({source}) : `{masked_key}`")


# 🚀 Bouton d'envoi de la question
if st.button("📤 Poser la question") and question:
    with st.spinner("🤖 Le modèle réfléchit..."):
        try:
            result = ask_question(
                question=question,
                model_name=model,
                temperature=temperature,
                k=k,
                user_api_key=user_api_key  # 👈 AJOUT ICI
                )

            # 🔸 Layout en deux colonnes
            col_left, col_right = st.columns([2, 1])

            # 🔸 Colonne gauche : réponse
            with col_left:
                st.success("✅ Réponse :")
                st.markdown(result["result"])

            # 🔸 Colonne droite : documents sources
            with col_right:
                st.markdown("""
                #### 📄 Sources documentaires utilisées
                """)
                if result.get("source_documents"):
                    for i, doc in enumerate(result["source_documents"], start=1):
                        title = doc.metadata.get("source", "Document inconnu")
                        page = doc.metadata.get("page", "?")
                        content = doc.page_content[:500] + "..."

                        with st.expander(f"📄 {title} (page {page})"):
                            st.markdown(content)
                else:
                    st.info("Aucune source documentaire n'a été retournée.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
            st.code(traceback.format_exc(), language="python")


# 📁 Téléversement de fichiers temporaires
st.warning("⚠️ Ce document ne sera **pas sauvegardé**. Il est utilisé uniquement pendant cette session.")
uploaded_files = st.file_uploader("📎 Téléverse un fichier PDF pour faire du RAG temporaire :", 
                                   type=["pdf"], 
                                   accept_multiple_files=True)

# 📄 Extraction des documents uploadés
session_docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Charger avec LangChain
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Ajouter la source pour affichage futur
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        session_docs.extend(docs)

        # Supprimer le fichier temporaire une fois chargé
        os.remove(tmp_path)

    st.success(f"✅ {len(session_docs)} page(s) PDF chargée(s) depuis les documents uploadés.")

if session_docs:
    st.divider()
    st.markdown("## 💬 Pose ta question sur le document uploadé")

    question = st.text_input("❓ Ta question (document temporaire)")

    if st.button("📤 Interroger le document") and question:
        with st.spinner("💡 Génération en cours..."):
            try:
                result = run_rag_on_uploaded_docs(
                    docs=session_docs,
                    question=question,
                    model=model,
                    temperature=temperature,
                    k=k,
                    user_api_key=user_api_key
                )

                st.success("🧠 Réponse :")
                st.markdown(result["result"])

                st.markdown("📎 **Sources :**")
                for doc in result["source_documents"]:
                    st.markdown(f"- `{doc.metadata['source']}`")

            except Exception as e:
                st.error(f"❌ Erreur pendant l'exécution : {e}")


# 🎡 Footer
st.markdown(
    "<div style='text-align: center; padding-top: 2rem;'>"
    "<sub>🚀 Créé avec 💪 par <b>Charif EL JAZOULI</b> • "
    "<a href='https://github.com/ton-lien-github' target='_blank'>GitHub</a></sub>"
    "</div>",
    unsafe_allow_html=True
)