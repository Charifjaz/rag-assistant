if __name__ == "__main__":
    import argparse
    from app import config
    from app.rag_engine import OpenAILLM, FAISSRetriever, RAGPipeline

    parser = argparse.ArgumentParser(
        description="Poser une question à ton assistant RAG basé sur OpenAI"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        help="Nom du modèle OpenAI (ex: gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Température du modèle (0.0 = déterministe, 1.0 = créatif)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.DEFAULT_K,
        help="Nombre de documents pertinents à récupérer",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        help="Question à poser (sinon posée en interactif)",
    )

    args = parser.parse_args()

    # 1. Récupération de la question
    question = args.question or input("❓ Pose ta question : ")

    # 2. Construction du pipeline
    retriever = FAISSRetriever(persist_path="vectorstore")
    llm = OpenAILLM(model_name=args.model, temperature=args.temperature)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    # 3. Exécution
    result = pipeline.ask(question, k=args.k)

    # 4. Affichage CLI
    print("🧠 Réponse :\n", result["result"])
    print("\n📄 Sources :")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Sans source"))
