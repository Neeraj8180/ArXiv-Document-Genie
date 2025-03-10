import streamlit as st
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS 
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from google.api_core import exceptions
import os
import time

def validate_api_key(api_key):
    """Validate basic API key format and accessibility"""
    try:
        if not api_key.startswith("AIza"):
            raise ValueError("Invalid API key format. Google API keys start with 'AIza'")
            
        genai.configure(api_key=api_key)
        models = genai.list_models()
        if "models/gemini-pro" not in [m.name for m in models]:
            raise PermissionError("Gemini Pro model not accessible. Enable API in Google Cloud Console.")
        return True
    except Exception as e:
        st.error(f"API Validation Failed: {str(e)}")
        return False

def get_arxiv_text(arxiv_id):
    """Fetch and validate ArXiv paper"""
    try:
        loader = ArxivLoader(query=arxiv_id)
        docs = loader.load()
        if not docs:
            raise ValueError("No paper found with this ID")
        title = docs[0].metadata.get("title", "this paper")
        return "\n".join([doc.page_content for doc in docs]), title
    except Exception as e:
        st.error(f"ArXiv Error: {str(e)}")
        return None, None

def get_text_chunks(text):
    """Safe text splitting"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, 
            chunk_overlap=1000,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Text Processing Error: {str(e)}")
        return []

def get_vector_store(text_chunks, api_key):
    """Create FAISS store with validation"""
    try:
        if not text_chunks:
            raise ValueError("No text chunks to process")
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Vector Store Error: {str(e)}")
        return False

def get_conversational_chain(api_key):
    """Create conversation chain with enhanced validation"""
    try:
        genai.configure(api_key=api_key)  # Critical configuration
        
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer isn't in the context, summarize the document clearly.
        
        Context:\n {context}\n
        Question: \n{question}\n
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=api_key,
            max_retries=3,
            timeout=30
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return StuffDocumentsChain(
            llm_chain=LLMChain(llm=model, prompt=prompt),
            document_variable_name="context"
        )
    except exceptions.NotFound:
        st.error("Model not found. Verify model name/API access.")
    except Exception as e:
        st.error(f"Chain Creation Error: {str(e)}")
    return None

def get_default_response(user_question, arxiv_title):
    """Predefined responses with case insensitivity"""
    normalized_input = user_question.lower().strip()
    greetings = {
        "hi": f"Hi! Ask me anything about '{arxiv_title}'",
        "hello": f"Hello! Let's discuss '{arxiv_title}'",
        "thank you": "You're welcome!",
        "bye": "Goodbye! Come back with more questions!",
        "history": "\n".join([f"**Q:** {q}\n**A:** {a}" for q, a in st.session_state.chat_history]),
    }
    return greetings.get(normalized_input, None)

def user_input():
    """Process user input with comprehensive error handling"""
    user_question = st.session_state.user_input.strip()
    if not user_question:
        return

    with st.spinner("Analyzing..."):
        try:
            # Check API validity first
            if not validate_api_key(st.session_state.api_key):
                return
                
            # Handle predefined responses
            if default_response := get_default_response(user_question, st.session_state.arxiv_title):
                st.session_state.chat_history.append((user_question, default_response))
                return

            # Process technical questions
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=st.session_state.api_key
            )
            
            vector_store = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            docs = vector_store.similarity_search(user_question, k=5)
            if not docs:
                docs = [Document(
                    page_content="\n".join([doc.page_content for doc in vector_store.docstore._dict.values()]),
                    metadata={"source": "Full Paper"}
                )]

            chain = get_conversational_chain(st.session_state.api_key)
            if chain:
                response = chain.invoke(
                    {"input_documents": docs, "question": user_question},
                    config={"max_retries": 2, "timeout": 20}
                )
                st.session_state.chat_history.append((user_question, response["output_text"]))
                
        except exceptions.PermissionDenied:
            st.error("API key rejected. Check key permissions.")
        except exceptions.DeadlineExceeded:
            st.error("Request timed out. Try a simpler question.")
        except Exception as e:
            st.error(f"Processing Error: {str(e)}")
            st.session_state.chat_history.append((user_question, "Failed to process question"))

    st.session_state.user_input = ""

def main():
    """Streamlit UI with guided setup"""
    st.set_page_config(page_title="ArXiv Genius", layout="wide")
    
    # Session state initialization
    for key in ["chat_history", "conversation_started", "api_key", "arxiv_id"]:
        if key not in st.session_state:
            st.session_state[key] = "" if key == "api_key" else [] if key == "chat_history" else False

    if not st.session_state.conversation_started:
        st.markdown("""
            ## 📄 ArXiv Genius: Research Paper Explorer
            **How to Use:**
            1. **Get API Key**: [Create Google API key](https://makersuite.google.com/app/apikey)
            2. **Find Paper ID**: Use numbers from arXiv URL (e.g. `2303.18223`)
            3. **Ask Questions**: About methods, results, or summaries
            """)
            
        st.session_state.api_key = st.text_input("🔑 Google API Key:", type="password")
        st.session_state.arxiv_id = st.text_input("📄 arXiv Paper ID:")
        
        if st.button("🚀 Process Paper"):
            if not validate_api_key(st.session_state.api_key):
                return
                
            with st.spinner("🧠 Processing paper..."):
                raw_text, title = get_arxiv_text(st.session_state.arxiv_id)
                if not raw_text:
                    return
                    
                chunks = get_text_chunks(raw_text)
                if not chunks:
                    st.error("Failed to process paper text")
                    return
                    
                if get_vector_store(chunks, st.session_state.api_key):
                    st.session_state.arxiv_title = title
                    st.session_state.conversation_started = True
                    st.rerun()
                    
    else:
        st.header(f"Chatting about: {st.session_state.arxiv_title}")
        st.text_input("💬 Ask about the paper...", key="user_input", on_change=user_input)
        
        with st.expander("📜 Chat History"):
            for q, a in st.session_state.chat_history:
                st.markdown(f"**Q:** {q}  \n**A:** {a}")
                
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
