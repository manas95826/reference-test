import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

import os
os.environ["OPENAI_API_KEY"]= os.gentenv('OPENAI_API_KEY')
# Initialize the embeddings and model
embd = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Define the Streamlit app
st.title("Text File Question-Answering with History")
st.subheader("Upload a text file and ask questions. The app will maintain a conversation history.")

# File upload section
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    # Load and split the text file
    text_loader = TextLoader(uploaded_file)
    document = text_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(document)
    
    # Create a vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="conversation_history",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever()
    
    # Initialize the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Question-answering section
    query = st.text_input("Ask a question:")
    
    if query:
        result = qa_chain({"query": query})
        answer = result["result"]
        st.session_state.conversation_history.append((query, answer))
        
        # Display the current answer
        st.write("**Answer:**", answer)
        
        # Display conversation history
        st.subheader("Conversation History")
        for idx, (q, a) in enumerate(st.session_state.conversation_history, 1):
            st.write(f"**Q{idx}:** {q}")
            st.write(f"**A{idx}:** {a}")
          
