from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from transformers import AutoModelForCausalLM
import os

directory_path = "https://github.com/husickova/keboola/tree/ffc8723ac5c0a71cb64666e3e0f0630b8120bd0f/soubory"
loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
data = loader.load()

#Step 05: Split the Extracted Data into Text Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

#Step 06:Downlaod the Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Step 08: Create Embeddings for each of the Text Chunk
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

#Import Model
llm = LlamaCpp(
    streaming = True,
    model_path = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto"),
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}))

query = "Based on the information you have, who would be strong candidates for open positions at Keboola and why?"

qa.invoke(query)

import sys

while True:
  user_input = input(f"Input Prompt: ")
  if user_input == 'exit':
    print('Exiting')
    sys.exit()
  if user_input == '':
    continue
  result = qa({'query': user_input})
  print(f"Answer: {result['result']}")

import streamlit as st

prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")

st.title("Keboola Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
