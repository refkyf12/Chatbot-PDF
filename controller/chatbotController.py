from langchain_community.document_loaders import PyPDFLoader
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

LANGCHAIN_API_KEY = 'use langsmith api key here'
LANGCHAIN_TRACING_V2 ="true"
os.environ["GROQ_API_KEY"] = "use groq api key here"

chunk_size = 1000
chunk_overlap = 200
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = ChatGroq(model="llama3-8b-8192")

def pdfLoader(filepath):
    async def load_pages():
        loader = PyPDFLoader(filepath)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        return pages

    return asyncio.run(load_pages())

def storeToVectorDB(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(pages)
    
    persist_directory = "vector_db"
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    
    vectorstore.persist()

    return vectorstore

def loadVectorDB():
    persist_directory = "vector_db"  # Direktori tempat vector store disimpan
    
    # Memuat vector store yang sudah ada
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    return vectorstore

def askQuestion(question, vectorDB):
    # Optional: Modify the question logic here if needed
    # In the example, we are keeping the placeholder question for simplicity
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context} Question: {question} Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorDB.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Get the result from the question
    result = qa_chain({"query": question})

    # Return the result
    return result["result"]