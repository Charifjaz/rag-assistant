import sys
import os
# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from langchain_community.document_loaders import PyPDFLoader
import tempfile
import streamlit as st
import traceback
from app import config
from app.utils.utils import load_api_key  
from app.utils.utils_streamlit import display_model_config
from app.rag_engine import RAGPipeline, FAISSRetriever, TemporaryFAISSRetriever,OpenAILLM



# ... (les imports restent identiques)

# 🎨 Configuration de la page
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🌟 Titre principal
st.title("🤖 RAG Assistant")
st.caption("Pose tes questions sur tes documents - Permanent ou Temporaire")

# 🧠 Barre latérale : Configuration
with st.sidebar:
    # ⚙️ Paramètres du modèle
    model, temperature, k = display_model_config("global")
    
    # 🔑 Clé API
    user_api_key = st.text_input(
        "🔑 Clé OpenAI (optionnelle)",
        type="password",
        placeholder="sk-...",
    )
    st.divider()
    
# 🔀 Création des onglets
tab1, tab2 = st.tabs(["📚 Base permanente", "📄 Session temporaire"])

# ==============================================================================
# ONGLET 1 - BASE PERMANENTE
# ==============================================================================
with tab1:
    col_query, col_result = st.columns([1, 3])
    
    with col_query:
        question = st.text_input(
            "❓ Question sur la base permanente",
            placeholder="Pose ta question ici...",
            key="main_question"
        )
        
        if st.button("🔍 Analyser", key="main_ask_btn") and question:
            with st.spinner("Recherche dans la base permanente..."):
                try:
                    # Initialisation et traitement
                    llm = OpenAILLM(model_name=model, temperature=temperature, user_api_key=user_api_key)
                    retriever = FAISSRetriever(persist_path=config.VECTORSTORE_PATH)
                    pipeline = RAGPipeline(retriever=retriever, llm=llm)
                    result = pipeline.ask(question, k=k)

                    # Affichage résultat
                    with col_result:
                        st.success("📝 Réponse :")
                        st.markdown(result["result"])
                        
                        st.subheader("🔎 Sources", divider="gray")
                        for doc in result.get("source_documents", []):
                            st.caption(f"📑 {doc.metadata.get('source', '')} (page {doc.metadata.get('page', '?')})")

                except Exception as e:
                    st.error(f"Erreur : {str(e)}")

# ==============================================================================
# ONGLET 2 - SESSION TEMPORAIRE
# ==============================================================================
with tab2:
    # 📤 Upload de fichiers
    uploaded_files = st.file_uploader(
        "Téléverser PDF(s) temporaire(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="temp_uploader"
    )
    
    # 📄 Traitement des fichiers
    if uploaded_files:
        session_docs = []
        
        # Extraction des pages
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                session_docs.extend(docs)
                os.remove(tmp.name)
        
        st.toast(f"✅ {len(session_docs)} pages chargées", icon="📄")
        
        # ❓ Question temporaire
        temp_question = st.text_input(
            "❓ Question sur le(s) document(s)",
            placeholder="Poser une question spécifique...",
            key="temp_question"
        )
        
        if st.button("🔍 Analyser documents", key="temp_ask_btn") and temp_question:
            with st.spinner("Analyse en cours..."):
                try:
                    # Traitement
                    llm = OpenAILLM(model_name=model, temperature=temperature, user_api_key=user_api_key)
                    retriever = TemporaryFAISSRetriever(docs=session_docs)
                    pipeline = RAGPipeline(retriever=retriever, llm=llm)
                    result = pipeline.ask(temp_question, k=k)

                    # Résultats
                    st.success("📝 Réponse :")
                    st.markdown(result["result"])
                    
                    st.subheader("🔎 Sources utilisées", divider="gray")
                    for doc in result.get("source_documents", []):
                        st.caption(f"📄 {doc.metadata['source']}")

                except Exception as e:
                    st.error(f"Erreur : {str(e)}")

# 🎡 Footer minimaliste
st.divider()
st.caption("🚀 Développé par [Votre nom] • [GitHub](https://github.com/ton-lien)")