from langchain_chroma import Chroma




def create_vector_store(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents = chunks,
        embeddings = embeddings,
        persist_directory = "data/vector_store"
    )


    return vector_store

