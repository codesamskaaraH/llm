import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS225896
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() # Loads .env file

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Loads API key
def get_pdf_text(pdf_docs):
    text = " "
    # Iterate through each PDF document path in the list
    for pdf in pdf_docs:
        # Create a PdfReader object for the current PDF document
        pdf_reader = PdfReader(pdf)
        # Iterate through each page in the PDF document
        for page in pdf_reader.pages:
            # Extract text from the current page and append it to the 'text' string
            text += page.extract_text()

    # Return the concatenated text from all PDF documents
    return text

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):     
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
def user_input(user_question):

    # replace this with sentence transformer
    # 
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("faiss_index", embeddings)

    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Print the response to the console
    print(response)

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("SupportVectors:BCamp01- 🐘🐘")

    user_question = st.text_input("Ask a Question after updating PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
