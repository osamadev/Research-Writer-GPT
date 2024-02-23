import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from streamlit import session_state
from langchain.prompts import PromptTemplate

pdf_folder_path = "data"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def process_pdf_documents(pdf_folder_path: str):
    documents = []

    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        length_function=len, 
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_documents = text_splitter.split_documents(documents)

    return chunked_documents


def load_or_persist_chromadb(collection_name: str, persist_directory: str) -> Chroma:
    client = chromadb.Client()

    existing_collections = client.list_collections()
    if existing_collections.count <= 0:
        print("Creating new collection and persisting documents.")
        client.create_collection(collection_name)
        chunked_documents = process_pdf_documents(pdf_folder_path)
        
        # Generate and persist the embeddings
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory, 
        )
        vectordb.persist()
    else:
        print("Collection already exists. Loading from ChromaDB.")
        vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=OpenAIEmbeddings())

    return vectordb

def create_agent_chain(vectordb):
    model_name = "gpt-4-0125-preview"
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer")

    template = """
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {chat_history}
        </hs>
        ------
        {question}
        Answer:
        """
    prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=template,
        )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever, 
        return_source_documents=True,
        condense_question_prompt=prompt
    )
    return qa_chain


def get_llm_response(query):
    collection_name = "research_papers"
    persist_directory = "vector_db"

    # Load embedded vectors from ChromaDB
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=OpenAIEmbeddings())
    chain = create_agent_chain(vectordb)
    response = chain(query)
    return response

def embed_documents():
    client = chromadb.Client()
    collection_name = "research_papers"
    persist_directory = "vector_db"

    # Re-checking for new files and re-embedding
    client.get_or_create_collection(collection_name)

    chunked_documents = process_pdf_documents(pdf_folder_path)
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    vectordb.persist()
    st.success("Documents re-embedded and Chroma index updated.")

## Parse results and cite sources
def process_llm_response(llm_response):
    search_results = llm_response['answer']
    sources = {}

    for source in llm_response["source_documents"]:
        doc_name = source.metadata['source'].replace('data\\', '')
        page_num = source.metadata['page']
        if doc_name not in sources:
            sources[doc_name] = {page_num}
        else:
            sources[doc_name].add(page_num)

    # Format the sources
    formatted_sources = [f"{doc} (Pages {' ,'.join(map(str, pages))})" for doc, pages in sources.items()]
    return search_results, formatted_sources

# Function to format sources as a bulleted list in Markdown
def format_sources_as_markdown(sources):
    formatted_sources = "\n".join([f"- {source}" for source in sources])
    return formatted_sources

# Function to list and delete PDF files
def list_and_manage_pdfs(pdf_folder_path):
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    selected_pdf = st.sidebar.selectbox("Select a PDF to delete", pdf_files)
    if st.sidebar.button(f"Delete '{selected_pdf}'"):
        os.remove(os.path.join(pdf_folder_path, selected_pdf))
        st.sidebar.success(f"Deleted {selected_pdf}")
        # Refresh the session state to update the file list
        session_state.update_file_list = True

# Function to upload new PDFs
def upload_pdf(pdf_folder_path):
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        with open(os.path.join(pdf_folder_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")
        # Refresh the session state to update the file list
        session_state.update_file_list = True


# Function to format and display search results elegantly
def display_search_results(results):
    if results:
        st.markdown("### Search Results")
        st.markdown(results)  # Assuming 'results' is a Markdown-compatible string
    else:
        st.markdown("No results found.")

# Streamlit App
st.set_page_config(page_title="Research Assistant RAG", page_icon="🌐", layout="wide")
st.header("🔎 Research Assistant RAG 🌐")

# Sidebar button for embedding documents
if st.sidebar.button("Embed Documents"):
    embed_documents()

# Sidebar for managing PDFs
with st.sidebar.expander("Manage PDFs"):
    list_and_manage_pdfs(pdf_folder_path)
    upload_pdf(pdf_folder_path)

if 'update_file_list' not in session_state:
    session_state.update_file_list = False

if session_state.update_file_list:
    # Refresh the page to update the file list
    st.rerun()

# Use a key for the text_input to reset the search when the text changes
form_input = st.text_input('Enter your query', key="query")

submit = st.button("Search")

def handle_submit():
    results, sources = process_llm_response(get_llm_response(form_input))
    display_search_results(results)
    formatted_sources = format_sources_as_markdown(sources)
    with st.expander("***Sources and Citation***"):
        st.markdown(formatted_sources)



if submit:
    handle_submit()


