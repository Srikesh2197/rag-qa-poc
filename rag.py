import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    TOP_K
)

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory='data\\chroma', embedding_function=embeddings)

def _get_llm(model_provider):
    if model_provider == "OpenAI":
        if OPENAI_API_KEY:
            return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)

def _build_chain(model_provider):
    vectorstore = get_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = _get_llm(model_provider)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the user's question into a standalone question using the chat history."
                       "If the question is already standalone, return it as-is."
                       "Do NOT answer the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=contextualize_q_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant for document Q&A. "
             "Answer ONLY using the provided context. "
             "If the answer isn't in the context, say you don't know.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def _to_langchain_messages(chat_history):
    msgs = []
    for user, ai in chat_history:
        msgs.append(HumanMessage(content=user))
        msgs.append(AIMessage(content=ai))
    return msgs

def ask(question, chat_history, model_provider="OpenAI"):
    chain = _build_chain(model_provider=model_provider)
    history_msgs = _to_langchain_messages(chat_history or [])

    result = chain.invoke({"input": question, "chat_history": history_msgs})

    return {
        "answer": result.get("answer", ""),
        "source_documents": result.get("context", []),
    }