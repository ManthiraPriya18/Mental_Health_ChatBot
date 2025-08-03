from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import shutil
import os

if os.path.exists("faiss_index"):
    shutil.rmtree("faiss_index")
    print("🧽 Old FAISS index deleted.")
# Step 2: Load PDF using PyMuPDFLoader
pdf_path = "MentalHealth_data.pdf"
loader = PyMuPDFLoader(pdf_path)
pages = loader.load()

# Step 3: Split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = splitter.split_documents(pages)

# Step 4: Embed using SentenceTransformer
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 5: Create FAISS vector store
db = FAISS.from_documents(docs, embedding)

# Step 6: Save the new FAISS index
db.save_local("faiss_index")
print("✅ New vector store created and saved as 'faiss_index/'")
