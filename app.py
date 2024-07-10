import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage
import os
import requests
 


def stream_data(query, remote_chain):
    '''Steaming output generator'''
    config = {"user_id": "user_id", "conversation_id": "conversation_id"}
    for r in remote_chain.stream(
        query,
        config={
                "configurable": config
        },
    ):
        yield r.content + ""
    
def main():
    st.title("🤗💬 Chain-Q/A")

    # Store chat contents and display messages from history on app rerun
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Stateful button configurations, a button's state stays True after click
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    def click_button():
        st.session_state.clicked = True

    # Sidebar contents, displays PDF uploader 
    with st.sidebar:
        st.markdown('''
        # Normal Q/A Mode ⭐
        Ask generic questions, seeking information, advice, or casual interaction.
        Chat-history is maintained to respond to questions with sufficient context.
        # Chat with PDF 💬
        Upload and analyze PDF documents. The chatbot can extract and interpret data, answering questions based on the content of the provided document.
        ''')
        placeholder = st.empty()
        btn = placeholder.button('Upload PDF document 📁', on_click=click_button, disabled=False, key='1')

    # CHAT WITH PDF
    if st.session_state.clicked:
        # nested widget will remain on the page until the clicked button's state is True
        
        #Disable the clicked button during PDF analysis
        placeholder.button('Upload PDF document 📁', disabled=True, key='2')

        def close_button():
            '''Hide nested widgets and enable PDF chat button once PDF analysis is completed'''
            st.session_state.clicked = False
            placeholder.button('Upload PDF document 📁', disabled=False, key='3')
        
        # Stop button and file uploader in sidebar
        with st.sidebar:
            st.button('Stop PDF chat.', on_click=close_button, type="primary")
            pdf = st.file_uploader("Upload your PDF", type='pdf')
 
        # Once PDF file is uploaded, read the contents
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Post the extracted content to langserve server for further chunking, splitting and vectordb dumping
            requests.post("http://localhost:8000/chunk/", json={"text":text})
            
            # Input questions for PDF Q/A
            if query:= st.chat_input("Ask questions from the PDF document ..."):
                # Display user message
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
                # Intialize runnable on rag_chain endpoint 
                remote_chain_rag = RemoteRunnable("http://localhost:8000/rag_chain/")

                # Invoke user query to the remote chain and yeild streaming output
                with st.chat_message("assistant"):
                    res = st.write_stream(stream_data(query, remote_chain_rag))
                # Display response
                st.session_state.messages.append({"role": "assistant", "content": res})
    # NORMAL Q/A
    else:
        # Take user input and display
        if query:= st.chat_input("Ask anything !"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            # Intialize runnable on q/a chain
            remote_chain_qa = RemoteRunnable("http://localhost:8000/chain/")
            # Invoke user query to the remote chain and yeild streaming output and display response
            with st.chat_message("assistant"):
                res = st.write_stream(stream_data({"question":query}, remote_chain_qa))
                st.session_state.messages.append({"role": "assistant", "content": res})
    
if __name__ == '__main__':
    main()