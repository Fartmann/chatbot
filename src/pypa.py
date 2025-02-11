import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize ChromaDB settings
settings = Settings(
    persist_directory="chroma_data",
    allow_reset=True
)
client = chromadb.Client(settings)

# Initialize or create a ChromaDB collection
def get_or_create_chroma_collection(name):
    try:
        return client.get_collection(name)
    except chromadb.errors.InvalidCollectionException:
        logging.info(f"Collection '{name}' not found. Creating new collection.")
        return client.create_collection(name)

chat_collection = get_or_create_chroma_collection("chat_history")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_texts" not in st.session_state:
    st.session_state.uploaded_texts = []  # List to store multiple uploaded files' content

# Function to store messages in ChromaDB
def store_in_chroma(role, content):
    chat_collection.add(
        documents=[content],
        metadatas=[{"role": role}],
        ids=[f"{role}-{len(st.session_state.messages)}"],
    )

# Function to retrieve stored messages from ChromaDB
def retrieve_from_chroma():
    try:
        results = chat_collection.get()
        return results["documents"], results["metadatas"]
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return [], []

# Function to stream chat response from Ollama
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

# Main Streamlit UI
def main():
    st.title("📢 Chat with LLMs Models using Ollama")
    logging.info("App started successfully.")

    # Sidebar model selection
    model = st.sidebar.selectbox("🛠 Choose Model", ["llama3.2:latest", "llama3.1 8b", "phi3", "mistral"])
    logging.info(f"Selected Model: {model}")

    # **FILE UPLOAD SECTION**
    uploaded_files = st.sidebar.file_uploader("📂 Upload text files", type=["txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                text = uploaded_file.read().decode("utf-8").strip()
                st.session_state.uploaded_texts.append(text)  # Store multiple files
                store_in_chroma("system", text)  # Store in ChromaDB
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
                logging.info(f"File {uploaded_file.name} uploaded successfully.")

                # Display uploaded content
                with st.expander(f"📄 View content of {uploaded_file.name}"):
                    st.text(text)

            except Exception as e:
                st.sidebar.error(f"Error loading file: {uploaded_file.name}")
                logging.error(f"Error reading {uploaded_file.name}: {e}")

    # **CHAT INPUT SECTION**
    if prompt := st.chat_input("💬 Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        store_in_chroma("user", prompt)
        logging.info(f"User input: {prompt}")

        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response...")

                with st.spinner("Thinking... 🤔"):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]

                        # Add uploaded files' text to the conversation context
                        if st.session_state.uploaded_texts:
                            context_text = "\n\n".join(st.session_state.uploaded_texts)
                            messages.insert(0, ChatMessage(role="system", content=f"Documents contain:\n{context_text}"))

                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\n🕒 Response Time: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        store_in_chroma("assistant", response_message_with_duration)
                        st.write(f"🕒 Response Time: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Time: {duration:.2f} seconds")

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating response.")
                        logging.error(f"Error: {str(e)}")

    # **SHOW SAVED CHAT HISTORY**
    if st.sidebar.button("📜 Show Chat History"):
        docs, metas = retrieve_from_chroma()
        st.sidebar.write("💬 Chat History:")
        for doc, meta in zip(docs, metas):
            st.sidebar.write(f"**{meta['role']}**: {doc}")

if __name__ == "__main__":
    main()
