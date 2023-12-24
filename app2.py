import logging
import streamlit as st
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from llama_index.llms import HuggingFaceLLM
from llama_index.llms import Gemini
from dotenv import load_dotenv
from llama_index.prompts.prompts import SimpleInputPrompt

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
#from huggingface_hub import login
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

st.title("RetailGPT")
with st.chat_message("user"):
    st.write("Hello USer 👋 : Please wait while we initialize RetailGPT !")

print("Here!!!")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "RetailGPT is ready! Shoot your questions"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Retail Documents ! This should take 1-2 minutes."):

        system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
        # This will wrap the default prompts that are internal to llama-index
        query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
        llm = GoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
        embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        service_context = ServiceContext.from_defaults(
          chunk_size=1024,
          llm=llm,
          embed_model=embed_model
        )
        #storage_context = StorageContext.from_defaults(persist_dir="resume_index")
        # Load index from the storage context
        #index = load_index_from_storage(storage_context=storage_context,service_context=service_context)
        documents = SimpleDirectoryReader("/content/Data/").load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        #index.storage_context.persist("resume_index")
        return index

index = load_data()
#chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)
#chat_engine = index.as_query_engine()
query_engine = index.as_query_engine()
#response = query_engine.query("Who is Paul Graham.")
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #response = chat_engine.chat(prompt)
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
