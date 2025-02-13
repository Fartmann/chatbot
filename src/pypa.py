import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

client = MongoClient("mongodb+srv://adamsatyshev10:skLZeav4NKdyvovk@cluster0.yumpv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["chat_db"]
chat_collection = db["chat_history"]

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_texts" not in st.session_state:
    st.session_state.uploaded_texts = []

# storing
def store_in_mongo(role, content):
    chat_collection.insert_one({"role": role, "content": content, "timestamp": time.time()})

# retrieving
def retrieve_from_mongo():
    try:
        results = chat_collection.find().sort("timestamp", 1)
        return [(doc["role"], doc["content"]) for doc in results]
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return []

# chat response
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        return "Error generating response."

# frontend
def main():
    st.title("Chat with Ollama")
    logging.info("App started successfully.")

    # sidebar
    model = st.sidebar.selectbox("Choose Model", ["llama3.2:latest", "llama3.1 8b"])
    logging.info(f"Selected Model: {model}")

    # upload files
    uploaded_files = st.sidebar.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                text = uploaded_file.read().decode("utf-8").strip()
                st.session_state.uploaded_texts.append(text)
                store_in_mongo("system", text)
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
                logging.info(f"File {uploaded_file.name} uploaded successfully.")

                # content display
                with st.expander(f"View content of {uploaded_file.name}"):
                    st.text(text)

            except Exception as e:
                st.sidebar.error(f"Error loading file: {uploaded_file.name}")
                logging.error(f"Error reading {uploaded_file.name}: {e}")

    # chat input
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        store_in_mongo("user", prompt)
        logging.info(f"User input: {prompt}")

        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response...")

                with st.spinner("Thinking..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]

                        # context of files
                        if st.session_state.uploaded_texts:
                            context_text = "\n\n".join(st.session_state.uploaded_texts)
                            messages.insert(0, ChatMessage(role="system", content=f"Documents contain:\n{context_text}"))

                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nResponse Time: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        store_in_mongo("assistant", response_message_with_duration)
                        st.write(f"Response Time: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Time: {duration:.2f} seconds")

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating response.")
                        logging.error(f"Error: {str(e)}")

    # chat history
    if st.sidebar.button("Show Chat History"):
        history = retrieve_from_mongo()
        st.sidebar.write("Chat History:")
        for role, content in history:
            st.sidebar.write(f"**{role}**: {content}")

if __name__ == "__main__":
    main()
