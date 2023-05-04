# PATH_TO_FILE = 'C:\langchain\sad.pdf'

import os


from streamlit_chat import message
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredFileIOLoader
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp

os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'fake_token'

PATH_TO_MODEL = 'fake_path_to_model'
PATH_TO_EMBS = 'fake_path_to_embs'
PATH_TO_FILE = 'fake_path_to_file'

import streamlit as st
import os
import pandas as pd
import numpy as np
from pathlib import Path

embeddings = SentenceTransformerEmbeddings(cache_folder = PATH_TO_EMBS)
llm  = LlamaCpp(model_path = PATH_TO_MODEL,n_ctx=2048,n_threads=12)

st.set_page_config(page_title="File Demo", page_icon=":robot:")
st.title('Question Answering App')

st.write('Upload a pdf file and ask questions about it')

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

uploaded_file = st.file_uploader("Choose a file", type=['pdf','txt'])



if uploaded_file is not None:
    ## save the pdf and then load it
    st.write('Uploaded file:')
    save_folder = PATH_TO_FILE
    save_path = Path(save_folder,uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    chat_history = []
    loader = UnstructuredFileLoader(uploaded_file.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents, embeddings)
    user_input = get_text()
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=False)
    if user_input:
        output = qa({"question": str(user_input), "chat_history": chat_history})
        ans = output['answer']
        st.session_state.past.append(user_input)
        st.session_state.generated.append(ans)
        chat_history.append(user_input)
        chat_history.append(ans)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
