import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
from chromadb import Client

# logging
logging.basicConfig(level=logging.INFO)

# chromadb initialization
try:
    chromadb_client = Client()
    collection = chromadb_client.get_or_create_collection("chat_responses")
except Exception as e:
    logging.error(f"ChromaDB initialization error: {str(e)}")
    collection = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

# function to do smth with model
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        for r in resp:
            response += r.delta
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except ConnectionError as ce:
        logging.error(f"Connection error: {str(ce)}")
        return "Connection failed. Please check if the server is running."
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

# function to add content to chromadb
def add_to_chroma_db(model, message):
    if not collection:
        st.error("ChromaDB is not initialized.")
        return
    try:
        if not message.get("content"):
            logging.error("No content to add to ChromaDB.")
            return
        collection.add(
            ids=[str(int(time.time()))],  # ID
            documents=[message["content"]],  # insides of the document
            metadatas=[{
                "model": model,
                "role": message["role"],
                "timestamp": int(time.time())
            }]
        )
        logging.info(f"Document added to ChromaDB: {message}")
    except Exception as e:
        logging.error(f"Error adding to ChromaDB: {str(e)}")

def handle_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            # add content
            add_to_chroma_db("file_upload", {"role": "user", "content": content})
            st.success(f"File '{uploaded_file.name}' has been processed and added.")
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload .txt files.")

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")

    # sidebar
    model = st.sidebar.selectbox("Choose a model", ["llama3.2", "llama3.1 8b", "phi3", "mistral"])
    logging.info(f"Model selected: {model}")

    # uploading files
    uploaded_files = st.file_uploader("Upload files (only .txt)", accept_multiple_files=True)
    if uploaded_files:
        handle_uploaded_files(uploaded_files)

    # interface
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        # message display
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Writing..."):
                    start_time = time.time()

                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time

                        # append
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})

                        # response display
                        st.write(response_message_with_duration)
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                        # response save
                        add_to_chroma_db(model, {"role": "assistant", "content": response_message_with_duration})
                    except Exception as e:
                        error_msg = f"An error occurred while generating the response: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.error(error_msg)
                        logging.error(error_msg)

    # checking chromadb
    if st.button("Check ChromaDB"):
        try:
            if collection:
                results = collection.get()
                st.write("Documents in ChromaDB:")
                for i, doc in enumerate(results["documents"]):
                    st.write(f"**ID**: {results['ids'][i]}")
                    st.write(f"**Content**: {doc}")
                    st.write(f"**Metadata**: {results['metadatas'][i]}")
                    st.write("---")
            else:
                st.error("ChromaDB is not initialized.")
        except Exception as e:
            st.error(f"Error checking ChromaDB: {str(e)}")
            logging.error(f"Error checking ChromaDB: {str(e)}")

    # clear chat
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared!")

if __name__ == "__main__":
    main()
