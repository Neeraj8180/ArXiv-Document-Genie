### 📋 README - ArXiv Document Genie  
### 🚀 Live App: ArXiv Document Genie
#### 📚 Overview  
ArXiv Document Genie is a **Streamlit-based chatbot** that allows users to fetch **research papers from ArXiv** and interact with them using **Google Gemini AI**. The system processes **ArXiv paper content**, stores it in a **FAISS vector database**, and enables **question-answering** using **retrieval-augmented generation (RAG).**  

---

## 🚀 Setup Instructions  

### **1⃣ Prerequisites**  
Ensure you have **Python 3.8+** installed.  

### **2⃣ Install Required Dependencies**  
Run the following command to install all dependencies:  
```bash
pip install streamlit langchain langchain_community faiss-cpu google-generativeai arxiv pymupdf
```

### **3⃣ Set Up Google API Key**  
- Obtain your **Google API Key** from [Google MakerSuite](https://makersuite.google.com/app/apikey).  
- You’ll need this key to use **Google Gemini AI** for question-answering.  

### **4⃣ Run the Streamlit App**  
Execute the following command:  
```bash
streamlit run Streamlit_App.py
```

### **5⃣ Usage**  
1⃣ Enter your **Google API Key** & **ArXiv Paper ID**  
2⃣ Click **Fetch & Process Paper**  
3⃣ Ask questions related to the paper  

---

## 📂 Project Structure  

| File | Description |
|------|------------|
| `Streamlit_App.py` | Main Streamlit application file |
| `requirements.txt` | List of dependencies |
| `README.md` | Documentation for setup and usage |

---

## 🛠 Technologies Used  
- **LangChain** (Document Processing, LLM Chaining)  
- **FAISS** (Vector Storage & Search)  
- **Google Generative AI (Gemini-Pro)**  
- **Streamlit** (Web Interface)  

---

## 📌 Short Write-Up on Retrieval & Generation  

This chatbot uses **Retrieval-Augmented Generation (RAG)** to **enhance LLM responses with external knowledge**.  

1⃣ **Retrieval Phase (FAISS Search):**  
   - The **ArXiv paper** is retrieved and **split into chunks** using **LangChain’s RecursiveCharacterTextSplitter**.  
   - These chunks are **converted into embeddings** using **Google Generative AI Embeddings**.  
   - The embeddings are **stored in FAISS**, a fast **vector database**.  
   - When a user asks a question, the **most relevant text chunks** are **retrieved** from FAISS.  

2⃣ **Generation Phase (LLM Response):**  
   - The retrieved document chunks are **passed to Google Gemini AI** for **answer generation**.  
   - A **custom prompt** ensures **accurate** and **context-aware** responses.  
   - If no relevant context is found, the chatbot **summarizes the entire document** instead.  

This **combination of retrieval (FAISS) and generation (LLM)** ensures that answers are **grounded in the ArXiv paper**, reducing **hallucination** while improving **accuracy**. 🚀  

---


