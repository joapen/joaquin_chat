# https://medium.com/@mcraddock/building-a-wardley-mapping-books-ai-assistant-using-langchain-faiss-and-openai-a-code-3f476d4f2626 

# 1.- Importing required packages
import os
import re
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain.templates import SystemMessagePromptTemplate

# 2.- Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 3.- paint left menu
st.set_page_config(page_title="Chat with joapen-GPT")
st.title("Chat with Joaquín-GPT")
st.sidebar.markdown("Developed by joapen (https://joapen.com)", unsafe_allow_html=True)
st.sidebar.markdown("LinkedIn Profile (https://www.linkedin.com/in/jpenafernand/)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 0.2")
st.sidebar.markdown("Not optimised")
st.sidebar.markdown("May run out of OpenAI credits")

# 4 Loading the datastore
DATA_STORE_DIR = "data_store"
st.write("Loading database")
vector_store = FAISS.load_local(DATA_STORE_DIR, OpenAIEmbeddings())

# 5.- Configuring the chat model
system_template="""Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

# 6.- set up the chat model with OpenAI
chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# 7.- Implementing the user interface and response
with st.spinner("Thinking..."):
    query = st.text_input("Question for Joaquín?", value="Has Joaquín written any book?")
    result = chain(query)

st.write("### Answer:")
st.write(result['answer'])
