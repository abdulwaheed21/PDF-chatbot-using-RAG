import streamlit as st
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Changed import here
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import openai
import os
from dotenv import load_dotenv
import logging
from typing import List
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration and Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_collection"
EMBEDDING_DIM = 3072
MODEL_NAME = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo-preview"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def setup_milvus() -> bool:
    """
    Set up Milvus connection and create collection if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            timeout=30
        )
        
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields=fields, description="Schema for RAG pipeline")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup Milvus: {str(e)}")
        return False

def process_pdf(file_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Process PDF file and return chunked documents.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add unique IDs to documents
        for i, doc in enumerate(chunks):
            doc.metadata['doc_id'] = f"doc_{i}"
            
        return chunks
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def initialize_rag_components(documents: List[Document]):
    """
    Initialize RAG components including embeddings and vector store.
    """
    try:
        embeddings = OpenAIEmbeddings(model=MODEL_NAME)
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            text_field="text",
            auto_id=False,
            primary_field="pk",
            vector_field="vector"
        )
        
        # Create a list of documents with explicit IDs
        docs_with_ids = []
        for doc in documents:
            docs_with_ids.append({
                "id": doc.metadata['doc_id'],
                "text": doc.page_content,
            })
        
        # Add documents with explicit IDs
        vector_store.add_texts(
            texts=[doc["text"] for doc in docs_with_ids],
            ids=[doc["id"] for doc in docs_with_ids]
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing RAG components: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="RAG Pipeline", layout="wide")
    st.title("PDF Chatbot using RAG Pipeline")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error(" OpenAI API Key not found in environment variables.")
        st.stop()
    openai.api_key = api_key
    
    # Initialize Milvus
    with st.spinner(' Connecting to Milvus...'):
        if not setup_milvus():
            st.error(" Failed to connect to Milvus. Please check if Milvus is running.")
            st.stop()
        st.success(" Connected to Milvus successfully!")
    
    # File upload
    uploaded_file = st.file_uploader(" Upload a PDF Document", type="pdf")
    if uploaded_file:
        with st.spinner(' Processing PDF...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Process documents
                documents = process_pdf(tmp_file_path)
                vector_store = initialize_rag_components(documents)
                
                st.success(f" Successfully processed {len(documents)} document chunks!")
                
                # Clean up temporary file
                os.remove(tmp_file_path)
                
                # Create QA chain with ChatOpenAI instead of OpenAI
                llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0.7)  # Changed here
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                
                # Question input
                user_question = st.text_input("Ask a question about the document:")
                if user_question:
                    with st.spinner(' Generating answer...'):
                        try:
                            result = qa_chain({"query": user_question})
                            
                            # Display answer
                            st.markdown("### Answer:")
                            st.write(result["result"])
                            
                            # Display source documents
                            with st.expander(" View Source Documents"):
                                for i, doc in enumerate(result["source_documents"], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.write(doc.page_content)
                                    st.markdown("---")
                                    
                        except Exception as e:
                            st.error(f" Error generating answer: {str(e)}")
                
            except Exception as e:
                st.error(f" Error processing document: {str(e)}")

if __name__ == "__main__":
    main()