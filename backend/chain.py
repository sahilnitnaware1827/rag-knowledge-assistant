from langchain_core.prompts import ChatPromptTemplate

def create_rag_chain(retriever, llm):

    prompt = ChatPromptTemplate(
        '''
            You are a helpful AI assistant.

            Answer the question using only the provided context.

            If the answer is not present in the context,
            say that you don't know based on the provided documents.

            Context:
            {context}

            Question:
            {question}

            Answer:
        '''
    )

    def rag_chain(question):

        documents = retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        messages = prompt.format_messages(
            context = context,
            question = question
        )

        response = llm.invoke(messages)

        return response.content

    return rag_chain

