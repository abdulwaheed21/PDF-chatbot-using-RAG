# **PDF Chatbot using RAG**  

A **Retrieval-Augmented Generation (RAG) pipeline** implemented using **Milvus, OpenAI, and Streamlit** to build a chatbot that can **answer questions** from PDF documents.  

## **🚀 Features**  
✅ Upload a **PDF document** and ask questions about its content  
✅ Uses **Milvus** for vector storage and **OpenAI GPT** for responses  
✅ Chunk-based retrieval with **LangChain** for better context  
✅ Streamlit-powered **interactive UI**  
✅ Fully **containerized** using Docker  

## **🛠️ Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/abdulwaheed21/PDF-chatbot-using-RAG.git
cd PDF-chatbot-using-RAG
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Environment Variables**  
Create a `.env` file and add your **OpenAI API key**:  
```ini
OPENAI_API_KEY="your-api-key-here"
```

### **4️⃣ Start Milvus (Vector Database) via Docker**  
```bash
docker-compose up -d
```

### **5️⃣ Run the Application**  
```bash
streamlit run app.py
```

## **📂 Project Structure**  
```
📦 PDF-chatbot-using-RAG
 ┣ 📜 .env                # API Key Configuration
 ┣ 📜 .gitignore          # Ignoring unnecessary files
 ┣ 📜 app.py              # Main Streamlit application
 ┣ 📜 requirements.txt    # Python dependencies
 ┣ 📜 docker-compose.yml  # Docker configuration for Milvus & MinIO
 ┣ 📜 LICENSE             # Project license
 ┗ 📜 README.md           # Documentation
```

## **📌 Technologies Used**  
- **Milvus** 🟢 (Vector Database for embeddings)  
- **LangChain** 🔗 (Chunking and retrieval framework)  
- **OpenAI API** 🤖 (LLM for chatbot responses)  
- **Streamlit** 🎨 (Interactive UI)  
- **Docker** 🐳 (Containerized deployment)  

## **📜 License**  
This project is licensed under the **MIT License**.  
