import streamlit as st
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS 
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document  # Import Document
import os
import time

def get_arxiv_text(arxiv_id):
    """Fetches the text content and title of a given ArXiv paper using its ID."""
    loader = ArxivLoader(query=arxiv_id)
    docs = loader.load()
    title = docs[0].metadata.get("title", "this paper")  # Default to "this paper" if title is missing
    return "\n".join([doc.page_content for doc in docs]), title

def get_text_chunks(text):
    """Splits the ArXiv paper text into smaller chunks for efficient processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    """Converts text chunks into vector embeddings and stores them in FAISS for retrieval."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    """Creates a conversational AI chain that retrieves and answers questions using an LLM."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, summarize the entire document in your own words as best as possible.
    
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key) #temperature introduced randomness 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Uses StuffDocumentsChain to properly process and answer questions from retrieved documents
    stuff_chain = StuffDocumentsChain(llm_chain=LLMChain(llm=model, prompt=prompt), document_variable_name="context")
    
    return stuff_chain

def get_default_response(user_question, arxiv_title):
    """Provides predefined responses for common greetings and previous prompt retrieval."""
    greetings = {
        "hi": f"Hi! It's my pleasure to chat with you. How can I help you with '{arxiv_title}'?",
        "hello": f"Hello! I'm here to assist you with '{arxiv_title}'. How can I help?",
        "thank you": "You're welcome! Feel free to ask anything about the paper.",
        "thankyou": "You're welcome! Feel free to ask anything about the paper.",
        "thanks": "You're welcome! Let me know if you have more questions.",
        "good morning": f"Good morning! How can I assist you with '{arxiv_title}'?",
        "good afternoon": f"Good afternoon! What would you like to know about '{arxiv_title}'?",
        "good evening": f"Good evening! Feel free to ask anything about '{arxiv_title}'.",
        "bye": "Goodbye! Have a great day and let me know if you need any help in the future.",
        "can you tell me again the answer of my previous prompts": "Certainly! Here’s a summary of our previous conversation:\n"
        + "\n".join([f"**You:** {q}\n**AI:** {a}" for q, a in st.session_state.chat_history]),
        "my previous prompts": "Certainly! Here’s a summary of our previous conversation:\n"
        + "\n".join([f"**You:** {q}\n**AI:** {a}" for q, a in st.session_state.chat_history]),
    }
    return greetings.get(user_question.lower().strip(), None)

def user_input():
    """Handles user queries, retrieves relevant documents, and generates responses."""
    user_question = st.session_state.user_input.strip()
    if not user_question:
        return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    default_response = get_default_response(user_question, st.session_state.arxiv_title)
    if default_response:
        st.session_state.chat_history.append((user_question, default_response))
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.session_state.api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)
        
        if not docs or user_question.lower() in ["detailed summary", "detailed summary of 200 words"]:
            raw_text = "\n".join([doc.page_content for doc in new_db.docstore._dict.values()]) 
            docs = [Document(page_content=raw_text, metadata={"source": "ArXiv"})]  # Wrap in Document object
        
        chain = get_conversational_chain(st.session_state.api_key)
        response = chain.invoke({"input_documents": docs, "question": user_question})
        st.session_state.chat_history.append((user_question, response["output_text"]))
    
    st.session_state.user_input = ""

def main():
    """Manages the Streamlit app interface and workflow for fetching and interacting with ArXiv papers."""
    st.set_page_config(page_title="ArXiv Document Genie", layout="wide")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    
    if not st.session_state.conversation_started:
        st.markdown("""
                ## ArXiv Document Genie: Get instant insights from Research Papers
                ## How to Use It:
                1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key [here](https://makersuite.google.com/app/apikey).
                2. **Enter an ArXiv Paper ID** (e.g., 2304.12345) and click **Fetch & Process Paper**.
                   - You can find ArXiv Paper ID as a unique **Numbers in url when you open any paper** or inside **arXiv webpage it is mentioned for each paper**.
                3. **Ask a Question**: Get instant answers based on the paper's content. And please use terms like '...from this paper' or '.....given paper' or '....fetched paper' to get closely related answers.
                """)

        st.session_state.api_key = st.text_input("Enter your Google API Key:", type="password")
        st.session_state.arxiv_id = st.text_input("Enter the ArXiv Paper ID:")
        
        if st.session_state.arxiv_id and st.button("Fetch and Process Paper") and st.session_state.api_key:
            with st.spinner("Fetching and Processing Paper..."):
                raw_text, st.session_state.arxiv_title = get_arxiv_text(st.session_state.arxiv_id)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, st.session_state.api_key)
                st.session_state.conversation_started = True
                st.rerun()
    else:
        st.sidebar.title("Chat History")
        for question, answer in st.session_state.chat_history:
            st.markdown(f"<b style='color:blue;'>You:</b> {question}", unsafe_allow_html=True)
            st.markdown(f"<b style='color:green;'>AI:</b> {answer}", unsafe_allow_html=True)
        
        st.text_input("Type your question and press Enter", key="user_input", on_change=user_input)

if __name__ == "__main__":
    main()