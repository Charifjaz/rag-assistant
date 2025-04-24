"""
mo7ami_diali_app.py – full single‑file Streamlit app
© 2025  •  Licence MIT
"""

# ----------------------------------------------------------------------------
# 1️⃣  Imports & setup
# ----------------------------------------------------------------------------
import sys, os, traceback
from pathlib import Path
import streamlit as st

# navbar helper (optional)
try:
    from streamlit_option_menu import option_menu
except ImportError:
    st.error("Le package 'streamlit-option-menu' est requis → pip install streamlit-option-menu")
    st.stop()

# local package path (adapt if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app import config
from app.utils.utils import load_api_key
from app.utils.utils_streamlit import display_model_config
from app.rag_engine import RAGPipeline, FAISSRetriever, OpenAILLM

# ----------------------------------------------------------------------------
# 2️⃣  Streamlit page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mo7ami Diali – Assistant Juridique IA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# 3️⃣  Custom CSS (kept minimal)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
html, body, [class*='st-'] { font-family: 'Inter', sans-serif; }

/* hero */
.hero { background: radial-gradient(circle at top left, #1e3c72, #2a5298 55%); padding:5rem 2rem 6rem; border-radius:1.5rem; color:#fff; text-align:center; }
.hero h1{font-size:3rem;margin-bottom:.5rem}
.hero p{font-size:1.25rem;opacity:.9}
.cta-btn{background:#ffd66e!important;color:#000!important;border:none!important;font-weight:600;padding:.75rem 2.25rem;border-radius:50px}

/* cards */
.card{background:#ffffff10;border:1px solid #ffffff22;padding:2rem;border-radius:1.25rem;text-align:center;transition:all .2s ease}
.card:hover{transform:translateY(-4px);box-shadow:0 4px 20px #00000040}
.pricing-title{font-size:1.5rem;font-weight:600;margin-bottom:.5rem}
.price{font-size:2.5rem;font-weight:700;margin:.5rem 0 1rem 0}
ul.features{list-style:none;padding:0;font-size:.95rem}
ul.features li{padding:.25rem 0}

footer{text-align:center;font-size:.8rem;margin-top:4rem;opacity:.7}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 4️⃣  Sidebar navigation
# ----------------------------------------------------------------------------
with st.sidebar:
    nav = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Consultation IA", "Tarifs", "À propos"],
        icons=["house", "chat-dots", "credit-card", "info-circle"],
        default_index=0,
    )

# helper: redirect between pages from button press
def redirect(page_name: str):
    st.session_state["_redirect"] = page_name

if "_redirect" in st.session_state:
    nav = st.session_state.pop("_redirect")

# ----------------------------------------------------------------------------
# 5️⃣  HOME PAGE
# ----------------------------------------------------------------------------
if nav == "Accueil":
    # hero section
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1>⚖️ Mo7ami Diali</h1>", unsafe_allow_html=True)
    st.markdown("<p>L’intelligence artificielle au service du droit marocain.</p>", unsafe_allow_html=True)
    if st.button("Poser ma question", key="cta", on_click=lambda: redirect("Consultation IA")):
        pass
    st.markdown("</div>", unsafe_allow_html=True)

    # features
    st.markdown("### Pourquoi Mo7ami Diali ?")
    cols = st.columns(3)
    features = [
        ("🧠", "Réponses instantanées", "GPT‑4 + RAG sur Codes marocains"),
        ("🔒", "Confidentialité totale", "Traitement local, aucun doc externe"),
        ("🎓", "Pour pros & étudiants", "Préparation d’actes, révisions d’examens"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(f"<div class='card'><div style='font-size:2rem'>{icon}</div><div class='pricing-title'>{title}</div><p>{desc}</p></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6️⃣  CONSULTATION IA PAGE
# ----------------------------------------------------------------------------
if nav == "Consultation IA":
    st.header("🤖 Consultation juridique assistée par IA")
    st.write("Posez votre question, l’IA cite les articles de loi marocains pertinents.")

    # advanced parameters
    with st.expander("Paramètres avancés du modèle"):
        model, temperature, k = display_model_config("global")
        user_api_key = st.text_input("🔑 Clé OpenAI (optionnelle)", type="password", placeholder="sk-...")

    # question input
    examples = [
        "Procédure de résiliation d’un bail commercial ?",
        "Conditions du redressement judiciaire ?",
        "Quels délais de paiement entre commerçants ?",
    ]
    idx = st.session_state.get("_ex_idx", 0)
    placeholder = examples[idx % len(examples)]
    st.session_state["_ex_idx"] = idx + 1

    question = st.text_input("📮 Votre question :", placeholder=placeholder)

    if st.button("📨 Obtenir la réponse") and question.strip():
        with st.spinner("Analyse en cours …"):
            try:
                load_api_key(user_api_key)
                llm = OpenAILLM(model_name=model, temperature=temperature, user_api_key=user_api_key)
                retriever = FAISSRetriever(persist_path=config.VECTORSTORE_PATH)
                pipeline = RAGPipeline(retriever=retriever, llm=llm)
                result = pipeline.ask(question, k=k)

                col_ans, col_src = st.columns([2, 1])
                with col_ans:
                    st.success("## 🧠 Réponse")
                    st.markdown(result["result"], unsafe_allow_html=True)
                with col_src:
                    st.markdown("## 📑 Sources")
                    if not result.get("source_documents"):
                        st.info("Aucune source retournée.")
                    else:
                        for doc in result["source_documents"]:
                            title = doc.metadata.get("source", "Document")
                            page = doc.metadata.get("page", "?")
                            with st.expander(f"{title} – p.{page}"):
                                st.markdown(doc.page_content[:900] + "…")
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.code(traceback.format_exc())

# ----------------------------------------------------------------------------
# 7️⃣  TARIFS PAGE
# ----------------------------------------------------------------------------
if nav == "Tarifs":
    st.header("💳 Nos offres")
    plans = [
        {"name":"Étudiant","price":"0 DH","features":["50 questions / mois","GPT‑3.5","Support e‑mail"]},
        {"name":"Pro Individuel","price":"149 DH / mois","features":["Illimité","GPT‑4 Turbo","Analyse PDF","Support 24 h"]},
        {"name":"Cabinet","price":"549 DH / mois","features":["5 comptes","GPT‑4 Turbo + Vision","Index privé","Support prioritaire"]},
    ]
    cols = st.columns(len(plans))
    for col, plan in zip(cols, plans):
        feats = "".join(f"<li>✅ {f}</li>" for f in plan["features"])
        col.markdown(
            f"<div class='card'><div class='pricing-title'>{plan['name']}</div><div class='price'>{plan['price']}</div><ul class='features'>{feats}</ul><button class='cta-btn'>S’abonner</button></div>",
            unsafe_allow_html=True,
        )
    st.markdown("**Besoin d’une offre sur-mesure ?** Contactez‑nous via l’onglet *À propos*.")

# ----------------------------------------------------------------------------
# 8️⃣  À PROPOS PAGE
# ----------------------------------------------------------------------------
if nav == "À propos":
    st.header("ℹ️ À propos de Mo7ami Diali")
    st.write("Projet open‑source visant à démocratiser l’accès au droit marocain grâce à l’IA.")

    st.subheader("L’équipe")
    st.markdown("* **Charif El Jazouli** – Lead dev IA\n* **Contributeurs GitHub** – vos PR sont les bienvenues !")

    st.subheader("Contactez‑nous")
    with st.form("contact"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Nom")
        email = col2.text_input("E‑mail")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Envoyer")
        if submitted:
            if email and message:
                st.success("Merci ! Nous vous répondrons sous 24 h.")
            else:
                st.error("Veuillez remplir les champs requis.")

# ----------------------------------------------------------------------------
# 9️⃣  Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<footer>⚖️ <b>Mo7ami Diali</b> © 2025 • Code sur <a href='https://github.com/ton-repo' target='_blank'>GitHub</a></footer>", unsafe_allow_html=True)
