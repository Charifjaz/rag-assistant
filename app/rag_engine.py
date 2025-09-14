# app/rag_components.py

import sys
import os
# 🔧 Ajout du dossier parent pour les imports depuis app/
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# === 0. Import Packages ===
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate


# === 1. Base Retriever ===
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int):
        pass


# === 2. FAISS Retriever ===
class FAISSRetriever(BaseRetriever):
    def __init__(self, persist_path: str = "vectorstore"):
        self.persist_path = persist_path
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = FAISS.load_local(
            persist_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int):
        return self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    

# === 2. TEMP FAISS Retriever ===
class TemporaryFAISSRetriever(BaseRetriever):
    def __init__(self, docs, chunk_size=500, chunk_overlap=50):
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        self.vectordb = FAISS.from_documents(chunks, embeddings)

    def retrieve(self, query: str, k: int):
        return self.vectordb.as_retriever(search_kwargs={"k": k})

# === 3. Base LLM ===
class BaseLLM(ABC):
    @abstractmethod
    def answer(self, question: str, documents: list):
        pass


# === 4. OpenAI LLM ===
class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float, user_api_key: str | None = None):
        # load_api_key(user_api_key)  # ← si tu veux permettre la clé custom
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        # -------- SYSTEM PROMPT --------------------------------------------------------
        self.system_prompt = """
            Vous êtes **Mo7ami Diali**, avocat virtuel inscrit fictivement au barreau de Casablanca.
            Votre mission : expliquer le droit marocain avec clarté, citer les textes applicables
            (Codes, dahirs, décrets…) et proposer les démarches pratiques.

            Répondez toujours :
            1. **Résumé** (≤ 2 phrases) – langage accessible.
            2. **Fondements juridiques** – citez chaque article invoqué (« Art. 62 C. commerce »).
            3. **Analyse détaillée** – raisonnement pas-à-pas.
            4. **Étapes / conseils** – actions concrètes.
            5. **Clause de non-responsabilité** : « Cette réponse est une information … ».
            Utilisez le français formel, vouvoyez l’utilisateur.
        """.strip()

        # -------- PROMPT TEMPLATE ------------------------------------------------------
        # {context} sera injecté automatiquement par RetrievalQA
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "user",
                    """
                        **Question :** {question}

                        Vous disposez des documents suivants :
                        {context}

                        Répondez en respectant la structure imposée.
                    """,
                ),
            ]
        )

    # -------------------------------------------------------------------------
    def answer(self, question: str, retriever):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt,
                "document_variable_name": "context"   # <– {context} dans le prompt
            },
            # 🔑 on dit à la chaîne que le champ d’entrée s’appelle « question »
            input_key="question",
        )
        # Le paramètre attendu par RetrievalQA est `question`, pas `query`
        return qa_chain.invoke({"question": question})
    
# === 5. RAG Pipeline ===
class RAGPipeline:
    def __init__(self, retriever: BaseRetriever, llm: BaseLLM):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, k: int = 3):
        retriever = self.retriever.retrieve(question, k)
        result = self.llm.answer(question, retriever)
        return {
            "result": result["result"],
            "source_documents":  result.get("source_documents", [])
        }
