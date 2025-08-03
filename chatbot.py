# chatbot.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM setup
llm = Ollama(model="llama3:8b")

# Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and supportive mental health assistant. Answer kindly and clearly.
If the query doesn't match the document, use your own knowledge to answer.

Context:\n{context}
Question:\n{question}
Helpful Answer:
"""
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

def get_response(user_input):
    result = qa_chain({"question": user_input})
    return result["answer"]

