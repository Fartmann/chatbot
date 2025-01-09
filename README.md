# Chatbot made by Adam Satyshev IT-2302

# Overview
This project provides a real-time chat application built in Python that allows users to interact with a Large Language Model (LLM) through a web browser. The application uses Streamlit for the front-end interface and integrates LLM called Ollama to generate responses. All chat queries and responses will be stored in a vector store for retrieval and analysis using a database called ChromaDB.

# Installation

1. Copy the repository to the needed directory :
```bash
git clone https://github.com/Fartmann/chatbot.git
```
...or download the ZIP-file and put the files in the needed directory.

2. Create a virtual environment and activate it:
```bash
python3 -m venv env
source .\env\Scripts\activate  # On Linux: env/bin/activate
```
3. Install the required dependencies:
```bash
pip install ollama chromadb streamlit
```
4. Choose the directory where your pypa.py is.

5. Run the application in the terminal
```bash
streamlit run pypa.py
```
P.S. You may have to install Ollama on your PC, but not through pip, but manually install it on its website https://ollama.com. And you have to run Ollama to run the program.

# Usage

Open the browser and navigate to the link that was written in terminal.

Choose the needed version on your right.

Enter your queries in the chat interface and receive responses from the integrated LLM.

All queries and responses are automatically stored in the vector store for future reference.

# Examples
1. - When is the independency day of Kazakhstan

- The Independence Day of Kazakhstan is December 16th.
Duration: 3.08 seconds

2. - What are the main characteristics of LLM

- Large Language Models (LLMs) are a type of artificial intelligence (AI) designed to process and generate human-like language. The main characteristics of LLMs include:

Scalability: LLMs can handle large amounts of text data, making them suitable for applications that require processing vast amounts of information.
Complexity: LLMs use complex neural network architectures, such as transformer models, to process and generate language.
Contextual understanding: LLMs can understand the context of a given text, including nuances like sarcasm, idioms, and figurative language.
Language generation: LLMs can generate coherent and natural-sounding text, often indistinguishable from human-written content.
Pre-training: Many LLMs are pre-trained on large datasets, which enables them to learn patterns and relationships in language.
Fine-tuning: LLMs can be fine-tuned for specific tasks or domains, allowing them to adapt to new contexts and applications.
Parallelization: LLMs can process multiple input sequences in parallel, making them well-suited for large-scale natural language processing (NLP) tasks.
Some notable examples of LLM characteristics include:

Self-attention mechanisms: LLMs use self-attention mechanisms to weigh the importance of different words or phrases within a sentence.
Contextualized embeddings: LLMs often use contextualized embeddings, which provide more nuanced representations of words and their relationships in language.
Hierarchical representations: LLMs can represent text at multiple levels of granularity, from words to sentences to documents.
Some common applications of LLMs include:

Language translation
Text summarization
Question answering
Chatbots and conversational AI
Content generation (e.g., articles, social media posts)
Sentiment analysis and opinion mining
Duration: 7.39 seconds
