import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

# import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from streamlit import session_state
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import Pinecone
from streamlit_extras.switch_page_button import switch_page

pdf_folder_path = "users_data"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import os
import pandas as pd
from uuid import uuid4
from tqdm.auto import tqdm

def process_pdf_documents(pdf_folder_path: str):
    records = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                records.append({
                    'title': os.path.splitext(file)[0],  # File name without extension as title
                    'context': doc,  # Storing the document content in 'context',
                    'file_name': file
                })

    return pd.DataFrame(records)

def split_and_embed_documents(dataframe, embed, index):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        length_function=len, 
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )
    
    batch_size = 100
    for i in tqdm(range(0, len(dataframe), batch_size)):
        batch_df = dataframe.iloc[i:i + batch_size]

        # first get metadata fields for this record
        metadatas = [{
            'title': record['title'],
            'text': str(record['context']),
            'file_name': record['file_name']
            } for _, record in batch_df.iterrows()]
        
        # Splitting the documents into chunks
        batch_df['context_chunks'] = batch_df['context'].apply(lambda x: text_splitter.split_documents([x]))

        # Flatten the context for embedding
        flattened_contexts = batch_df.explode('context_chunks')['context_chunks'].tolist()

        # Ensure each element in flattened_contexts is a string
        flattened_contexts = [str(context) for context in flattened_contexts if context]

        # Check if embed_documents expects a list or single strings
        try:
            # If it expects a list
            embeds = embed.embed_documents(flattened_contexts)
        except TypeError:
            # If it expects individual strings
            embeds = [embed.embed_documents(context) for context in flattened_contexts]

        # Create IDs for each chunk
        ids = [str(uuid4()) for _ in range(len(embeds))]

        # Upsert to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadatas))

def embed_documents():
    pdf_folder_path = './users_data/'
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    # Check if there are PDF files in the folder
    if not pdf_files:
        st.error("No PDF files found in the folder. Please upload documents to proceed.")
        return

    index = initialize_pinecone()
    data = process_pdf_documents(pdf_folder_path)

    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    
    split_and_embed_documents(data, embed, index)

    # After embedding, delete the PDF files
    for filename in pdf_files:
        os.remove(os.path.join(pdf_folder_path, filename))
    
    # Display a success message
    st.success("Document embeddings completed successfully.")

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
    uploaded_files = st.sidebar.file_uploader("Upload Research Papers", type="pdf", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
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
        st.markdown(results)  
    else:
        st.markdown("No results found.")

# Initialize session state for conversation history if it does not exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


def handle_submit(query):
    results = get_llm_response(query)
    # Append results to the conversation history
    st.session_state.conversation_history.append((query, results))
    display_search_results(results)

def setup_ui():
    st.set_page_config(page_title="Research Assistant RAG", page_icon="🎓", layout="wide")
    st.title("🎓 Research Assistant RAG 🔎")

    # Upload and Manage PDFs in Sidebar
    with st.sidebar:
        st.header("Manage Research Papers")
        upload_pdf(pdf_folder_path)
        # Button to trigger document embedding process
        if st.sidebar.button("Embed Documents"):
            embed_documents()

    with st.form("search_form"):
        query = st.text_input("Enter your query", help="Type your research query here.")
        submit = st.form_submit_button("Search")

    if submit:
        if query:
            handle_submit(query)
        else:
            st.error("Please enter your question first.")

    # Display conversation history
    with st.expander("Conversation History"):
        for i, (q, a) in enumerate(st.session_state.conversation_history, start=1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")

# Main App
def main():
    setup_ui()

if __name__ == "__main__":
    if ("authentication_status" not in st.session_state or
            st.session_state["authentication_status"] is None or
            st.session_state["authentication_status"] is False):
        switch_page("Home")

    if st.session_state["authentication_status"]:
        main()


