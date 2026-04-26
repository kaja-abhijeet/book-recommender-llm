from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load raw text
raw_docs = TextLoader("tagged_description.txt", encoding="utf-8").load()

# 🔥 IMPORTANT: no splitting
documents = []

for i, doc in enumerate(raw_docs):
    documents.append(
        Document(
            page_content=doc.page_content,
            metadata={"index": i}   # matches CSV row
        )
    )

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create DB
db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ DB CREATED CORRECTLY")