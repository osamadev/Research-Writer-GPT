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
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import Pinecone

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

def initialize_pinecone():
    index_name = "research-assistant" 

    from pinecone import Pinecone

    pc = Pinecone(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"]
        
    )
    # if index_name not in pc.list_indexes():
    #     pc.create_index(name=index_name, metric='cosine', dimension=1536, spec={})
    # Target the index
    index = pc.Index(index_name)
    return index

def create_coversational_chain():
    from langchain_community.chat_models import ChatOpenAI
    index = initialize_pinecone()
    model_name = 'text-embedding-ada-002'
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    openai_model = "gpt-4-0125-preview"

    text_field = "text"

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    retriever = vectorstore.as_retriever()
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name=openai_model, temperature=0.1)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def create_agent_chain():
    from langchain_community.chat_models import ChatOpenAI
    index = initialize_pinecone()
    model_name = 'text-embedding-ada-002'
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    openai_model = "gpt-4-0125-preview"

    text_field = "text"

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name=openai_model, temperature=0.1)

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
    chain = create_coversational_chain()
    response = chain.invoke(query)
    return response

# def embed_documents():
#     from langchain_community.document_loaders import PyPDFLoader

#     loader = PyPDFLoader('./dataset', glob="**/*.pdf")
#     data = loader.load()

#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     import tqdm

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, 
#             chunk_overlap=100, 
#             length_function=len, 
#             is_separator_regex=False,
#             separators=["\n\n", "\n", " ", ""]
#     )
#     docs_chunks = text_splitter.split_documents(data)

#     batch_size = 100
#     OPENAI_API_KEY =os.environ["OPENAI_API_KEY"]
#     model_name = 'text-embedding-ada-002'

#     embed = OpenAIEmbeddings(
#         model=model_name,
#         openai_api_key=OPENAI_API_KEY
#     )
#     texts = []
#     metadatas = []

#     for i in tqdm(range(0, len(docs_chunks), batch_size)):
#         # get end of batch
#         i_end = min(len(docs_chunks), i+batch_size)
#         batch = docs_chunks[i:i_end]

#         # first get metadata fields for this record
#         metadatas = [{
#             'text': record.page_content
#         } for _, record in enumerate(batch)]
#         # get the list of contexts / documents
#         documents = batch
#         # create document embeddings

#         embeds = embed.embed_documents([str(doc.page_content) for doc in documents])
#         # get IDs
#         # Create IDs for each chunk
#         ids = [uuid4().hex for _ in range(len(embeds))]
#         # add everything to pinecone
#         index.upsert(vectors=zip(ids, embeds, metadatas))
#         st.success("Documents re-embedded and Chroma index updated.")

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
    search_results = llm_response
    # sources = {}

    # for source in llm_response["source_documents"]:
    #     doc_name = source.metadata['source'].replace('data\\', '')
    #     page_num = source.metadata['page']
    #     if doc_name not in sources:
    #         sources[doc_name] = {page_num}
    #     else:
    #         sources[doc_name].add(page_num)

    # # Format the sources
    # formatted_sources = [f"{doc} (Pages {' ,'.join(map(str, pages))})" for doc, pages in sources.items()]
    return search_results

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
    uploaded_file = st.sidebar.file_uploader("Upload a research paper", type="pdf")
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
st.set_page_config(page_title="Research Assistant RAG", page_icon="🎓", layout="wide")
st.header("🎓 Research Assistant RAG 🔎")

# Sidebar button for embedding documents
if st.sidebar.button("Embed Documents"):
    embed_documents()

# Sidebar for managing PDFs
with st.sidebar.expander("Upload Research Papers"):
    #list_and_manage_pdfs(pdf_folder_path)
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
    results = process_llm_response(get_llm_response(form_input))
    display_search_results(results)
    # formatted_sources = format_sources_as_markdown(sources)
    # with st.expander("***Sources and Citation***"):
    #     st.markdown(formatted_sources)

if submit:
    handle_submit()


