import streamlit as st
from dotenv import load_dotenv #load .env file
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS #Facebook AI Similarity Search 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



#Coverting the Pdf to Text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
# Creating the chucks for Vectorstore by splitting into 1000 words and overlapping of 200 words
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# feeding the text chunks to openAI
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Getting the Response Back
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

#getting user response
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write( message.content)
        else:
            st.write( message.content)


def main():
    load_dotenv() #loading API Key
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.chat_message("user"):
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write(vectorstore)
                

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
if __name__ == '__main__':
    main()