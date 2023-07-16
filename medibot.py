# import libraries
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import openai
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Tool, initialize_agent
import streamlit as st

# load environment variables
load_dotenv()

# embeddings and persist vector database
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorDb = Chroma(persist_directory='./medical_chroma_db', embedding_function=embeddings, collection_name="MedicalInformation")


# create llm
llm = OpenAI(temperature=0.0)
chat_history=[]

# create the frontend
st.set_page_config(page_title="Bajaj Medibot", page_icon='./Bajaj_logo.png')
st.title("Hello World")

options = st.multiselect("Choose the sources:",['webmd','NHS','CDC'])
query = st.text_input("Enter Your query")


if query:
    sources=[]
    for i in range(len(options)):
        sources.append(options[i])
    bot_response=[]
    source_cites=[]

    for source in sources:
        search_kwargs = {"filter":{"source":source}}
        retrieverChain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorDb.as_retriever(search_kwargs=search_kwargs), return_source_documents=True)
        result = retrieverChain({"question": query, "chat_history": chat_history})
        bot_response.append(result['answer'])
        source_cites.append(result["source_documents"][0].metadata['url'])
    
    st.write(bot_response[0])
    with st.expander('Sources Citations'):
        for cite in source_cites:
            st.write(cite)