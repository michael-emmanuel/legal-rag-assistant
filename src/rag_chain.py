import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def get_legal_rag_chain():
    # 1. Load the vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    
    # 2. Define specialized Legal Prompt
    template = """You are a specialized legal assistant. Use the following pieces of context to answer the question. 
    If you don't know the answer, state that you don't know—do not make up legal facts.
    
    Context: {context}
    Question: {question}
    
    Helpful Legal Answer:"""
    
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # 3. Initialize LLM (Could point to a local LoRA-tuned model)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 4. Create RAG Chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
