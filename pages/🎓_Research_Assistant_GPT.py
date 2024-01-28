import os
import re
import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import Clarifai as Clarifaillm
from langchain.llms.openai import OpenAI
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun, SemanticScholarAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from helpers.images_analyzer_tool import ClarifaiImageAnalyzerFromURL, ClarifaiImageAnalyzerFromFile
from audio_recorder_streamlit import audio_recorder
from langchain.schema.messages import SystemMessage
from clarifai.client.model import Model
from langchain_community.vectorstores import Clarifai as Clarifaivectorstore
from langchain_community.llms import Clarifai as Clarifaillm
from langchain_community.tools import PubmedQueryRun
from helpers.academia_kb_tool import AcademiaKBTool
from helpers.CustomStreamingStdOutCallbackHandler import CustomStreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from typing import List
from langchain.tools import Tool
import threading


def transcribe(audio_file):
    client = OpenAI()
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    return transcript

# Load environment variables
# load_dotenv()

def init_app():
    os.environ["CLARIFAI_PAT"] = st.secrets["CLARIFAI_PAT"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["SERP_API_KEY"] = st.secrets["SERP_API_KEY"]

# Function to extract prompt and image URL from combined input
def extract_prompt_and_url(combined_prompt):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', combined_prompt)
    if urls:
        image_url = urls[0]
        prompt = combined_prompt.replace(image_url, '').strip()
    else:
        image_url = None
        prompt = combined_prompt

    return prompt, image_url

# Function to convert text to speech
def text_to_speech(text):
    inference_params = dict(voice="alloy", speed=1.0, api_key=os.getenv("OPEN_AI_KEY"))
    model_prediction = Model("https://clarifai.com/openai/tts/models/openai-tts-1").predict_by_bytes(text.encode(), input_type="text", inference_params=inference_params)
    output_base64 = model_prediction.outputs[0].data.audio.base64
    return output_base64

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

def pdf_to_text(file_streams) -> List[Document]:
    docs = []
    for file_stream in file_streams:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_stream)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs.extend(text_splitter.split_documents(documents))
        # Clean up the temporary file
        os.remove(temp_file_path)

    return docs

# Function to handle the conversation logic
def handle_conversation(user_input, chat_agent):
    if user_input:
        # Append user input to conversation history
        conversation_history = st.session_state.get('conversation_history', [])
        conversation_history.append(f"👤 You: {user_input}")
        # Process the input through the chat agent
        response = chat_agent()
        if 'output' in response:
            conversation_history.append(f"🤖 AI: {response['output']}")

        # Update the conversation history in session state
        st.session_state['conversation_history'] = conversation_history


# Initialize the Langchain agent
def initialize_chatbot():
    system_message = SystemMessage(
          content="""You are an expert researcher who can help other researchers from different domains to 
          find relevant resources and papers for their research domain. You have differnt tools in hand to get the best results 
          which includes Google Scolar, Semantic Scholar, PubMed articles and literatures, and Google Search. 
          You have to use these tools efficiently to find out answers, recommend papers and summarize articles based on 
          the researcher query to compile a detailed response which covers their questions. Make sure to use mutiple tools if required.
          Format the final answer in a well structured and organized way. Use APA style wherever needed.
          """
      )
    model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"

    # llm = Clarifaillm(model_url=model_url)

    llm = OpenAI(callbacks=[CustomStreamingStdOutCallbackHandler()] ,temperature=0 ,streaming=True)
    
    semantic_scholar_tool = SemanticScholarQueryRun(name="Semantic Scholar", 
                                                    api_wrapper=SemanticScholarAPIWrapper(top_k_results = 10, load_max_docs = 10),
                                                    description="""Use this tool to answer questions about sentific papers and literatures,
                                     or to provide recommendations or summaries about the papers or research topics. """, )
    
    google_scholar_tool = Tool("Google Scholar", func=GoogleScholarAPIWrapper().run, description="""Use this tool to help answering 
                                                reasrch questions and provide recommendations on the research papers and literatures.
                                                If the researcher asked for links to the papers or literatures, use this search tool first.""")

    pubmed_research_tool = PubmedQueryRun()
    
    arxiv_tool = Tool("Arxiv Articles and Literatures", 
                      ArxivAPIWrapper(top_k_results = 10,
            ARXIV_MAX_QUERY_LENGTH = 300,
            load_max_docs = 10,
            load_all_available_meta = False,
            doc_content_chars_max = 40000).run,
                      description="Use this tool to get details about articles published in Arxiv"
                      )

    image_analyzer_tool = ClarifaiImageAnalyzerFromURL()

    academia_kb_tool = AcademiaKBTool()

    tools = [semantic_scholar_tool, arxiv_tool, pubmed_research_tool, image_analyzer_tool]

    conversational_memory = ConversationBufferMemory(
        memory_key='chat_history', input_key='input', output_key='output', return_messages=True)

    chat_agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=False,
        early_stopping_method='generate',
        handle_parsing_errors=False,
        memory=conversational_memory,
        system_message = system_message,
        max_iterations=3, 
        kwargs= {
            "system_message": system_message
        }
    )
    return chat_agent

def generate_speech(text):
    try:
        return text_to_speech(text)
    except Exception as e:
        st.error(f"Error in text-to-speech generation: {e}")
        return None

def process_user_input(user_input, chat_agent):
    # Update conversation history with user query
    conversation_history = st.session_state['conversation_history']
    conversation_history.append({'type': 'text', 'content': f"👤 You: {user_input}"})
    st.session_state['conversation_history'] = conversation_history

    # Get response from llm
    response = chat_agent(st.session_state['conversation_history'])
    if 'output' in response:
        new_response = f"🤖 AI: {response['output']}"
        conversation_history.append({'type': 'text', 'content': new_response})
        st.session_state['conversation_history'] = conversation_history

        # Generate and store audio if enabled
        if st.session_state['enable_audio_transcription']:
            speech_audio = generate_speech(response['output'])
            if speech_audio:
                conversation_history.append({'type': 'audio', 'content': speech_audio})
                st.session_state['conversation_history'] = conversation_history

def main():
    st.set_page_config("Research Assistant GPT", layout="wide", page_icon="🎓")
    st.title("🎓 Research Assistant GPT 🤖")
    st.caption("Explore the world of academia with ease 🎓! Our AI-powered research assistant 🤖 is here to help you discover and recommend scholarly papers 📚, provide insightful summaries 📄, and guide you through a sea of literature with style and efficiency 🌟.")
        # Initialize chatbot
    chat_agent = initialize_chatbot()

     # PDF file uploader
    uploaded_files = st.sidebar.file_uploader("Upload PDF files to chat with or to get insights from", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        process_files_button = st.sidebar.button("Process Uploaded Files")
        if process_files_button:
            documents = [file.getvalue() for file in uploaded_files]
            docs = pdf_to_text(documents)

            Clarifaivectorstore.from_documents(
                user_id=st.secrets["USER_ID"],
                app_id=st.secrets["APP_ID"],
                documents=docs,
                number_of_docs=len(docs),
                pat=os.getenv("CLARIFAI_PAT")
            )
            st.sidebar.success("Documents processed. You can now start asking questions about the uploaded documents.")
            uploaded_files = None
    
    # Initialize session state for conversation history and audio transcription setting
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'enable_audio_transcription' not in st.session_state:
        st.session_state['enable_audio_transcription'] = False

    # Option to enable/disable audio transcription
    enable_audio_transcription = st.checkbox("Enable audio transcription of responses", value=st.session_state['enable_audio_transcription'])

    # Update session state only when there is a change
    if enable_audio_transcription != st.session_state['enable_audio_transcription']:
        st.session_state['enable_audio_transcription'] = enable_audio_transcription

    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None
    
    # Chat interface
    user_input = st.chat_input("Enter your research question here", key="chat_input")
    # Image Uploader
    uploaded_image = st.file_uploader("Optionally, upload an image for analysis", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.session_state['uploaded_image'] = uploaded_image

     # Record Audio
    audio_bytes = audio_recorder(text="Click the mic to start recording", icon_size="1x", pause_threshold=1.2)

    if user_input:
        process_user_input(user_input, chat_agent)

    if audio_bytes:
        # Process the recorded audio
        save_audio_file(audio_bytes=audio_bytes, file_extension="mp3")

        # Find the newest audio file
        def get_creation_time(file):
            return os.path.getctime(os.path.join("./temp/", file))

        audio_file_path = './temp/' + max(
            [f for f in os.listdir("./temp/") if f.startswith("audio")],
            key=get_creation_time
        )

        # Transcribe the audio file
        transcript_text = transcribe_audio(audio_file_path)
        st.session_state['conversation_history'].append(f"👤 You: {transcript_text}")

        # Get response from chatbot
        response = chat_agent(st.session_state['conversation_history'])

        if 'output' in response:
            new_response = f"🤖 AI: {response['output']}"
            st.session_state['conversation_history'].append(new_response)

            # Transcribe response to audio if enabled
            if st.session_state['enable_audio_transcription']:
                speech_audio = text_to_speech(response['output'])
                st.audio(speech_audio, format='audio/mp3', start_time=0)

        # Clear the recorded audio to allow new recording
        audio_bytes = None



    # Display conversation history in an expander
    with st.expander("**Conversation History**", expanded=True):
        for item in st.session_state['conversation_history']:
            if item['type'] == 'text':
                st.markdown(item['content'], unsafe_allow_html=True)
            elif item['type'] == 'audio':
                st.audio(item['content'], format='audio/mp3', start_time=0)


# Streamlit UI
def main2():
    st.set_page_config("Research Assistant GPT", layout="wide", page_icon="🎓")
    st.title("🎓 Research Assistant GPT 🤖")
    st.caption("Explore the world of academia with ease 🎓! Our AI-powered research assistant 🤖 is here to help you discover and recommend scholarly papers 📚, provide insightful summaries 📄, and guide you through a sea of literature with style and efficiency 🌟.")
    # Initialize chatbot
    chat_agent = initialize_chatbot()

     # PDF file uploader
    uploaded_files = st.sidebar.file_uploader("Upload PDF files to chat with or to get insights from", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        process_files_button = st.sidebar.button("Process Uploaded Files")
        if process_files_button:
            documents = [file.getvalue() for file in uploaded_files]
            docs = pdf_to_text(documents)

            Clarifaivectorstore.from_documents(
                user_id=st.secrets["USER_ID"],
                app_id=st.secrets["APP_ID"],
                documents=docs,
                number_of_docs=len(docs),
                pat=os.getenv("CLARIFAI_PAT")
            )
            st.sidebar.success("Documents processed. You can now start asking questions about the uploaded documents.")
            uploaded_files = None
    
    # Initialize session state for conversation history and audio transcription setting
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'enable_audio_transcription' not in st.session_state:
        st.session_state['enable_audio_transcription'] = False

    # Option to enable/disable audio transcription
    enable_audio_transcription = st.checkbox("Enable audio transcription of responses", value=st.session_state['enable_audio_transcription'])

    # Update session state only when there is a change
    if enable_audio_transcription != st.session_state['enable_audio_transcription']:
        st.session_state['enable_audio_transcription'] = enable_audio_transcription

    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None
    
    # Chat interface
    user_input = st.chat_input("Enter your research question here", key="chat_input2")
    # Image Uploader
    uploaded_image = st.file_uploader("Optionally, upload an image for analysis", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.session_state['uploaded_image'] = uploaded_image

     # Record Audio
    audio_bytes = audio_recorder(text="Click the mic to start recording", icon_size="1x", pause_threshold=1.2)

    if user_input:
        # Update conversation history with user query
        st.session_state['conversation_history'].append(f"👤 You: {user_input}")

        if st.session_state['uploaded_image'] is not None:
            clarifai_analyzer = ClarifaiImageAnalyzerFromFile()
            tool_input = {
                "file_bytes": st.session_state['uploaded_image'].getvalue(),
                "prompt": st.session_state['conversation_history']
            }
            response = clarifai_analyzer.run(tool_input, callbacks=[CustomStreamingStdOutCallbackHandler()])
            if response:
                new_response = f"🤖 AI: {response}"
                st.session_state['conversation_history'].append(new_response)

                # Transcribe response to audio if enabled
                if st.session_state['enable_audio_transcription']:
                    speech_audio = text_to_speech(response['output'])
                    st.audio(speech_audio, format='audio/mp3', start_time=0)
        else:
            # Get response from llm
            response = chat_agent(st.session_state['conversation_history'])
            if 'output' in response:
                new_response = f"🤖 AI: {response['output']}"
                st.session_state['conversation_history'].append(new_response)

                # Transcribe response to audio if enabled
                if st.session_state['enable_audio_transcription']:
                    speech_audio = text_to_speech(response['output'])
                    st.audio(speech_audio, format='audio/mp3', start_time=0)
        
        # Reset the file uploader widget
        st.session_state['uploaded_image'] = None
        uploaded_image = None
        

    if audio_bytes:
        # Process the recorded audio
        save_audio_file(audio_bytes=audio_bytes, file_extension="mp3")

        # Find the newest audio file
        def get_creation_time(file):
            return os.path.getctime(os.path.join("./temp/", file))

        audio_file_path = './temp/' + max(
            [f for f in os.listdir("./temp/") if f.startswith("audio")],
            key=get_creation_time
        )

        # Transcribe the audio file
        transcript_text = transcribe_audio(audio_file_path)
        st.session_state['conversation_history'].append(f"👤 You: {transcript_text}")

        # Get response from chatbot
        response = chat_agent(st.session_state['conversation_history'])

        if 'output' in response:
            new_response = f"🤖 AI: {response['output']}"
            st.session_state['conversation_history'].append(new_response)

            # Transcribe response to audio if enabled
            if st.session_state['enable_audio_transcription']:
                speech_audio = text_to_speech(response['output'])
                st.audio(speech_audio, format='audio/mp3', start_time=0)

        # Clear the recorded audio to allow new recording
        audio_bytes = None


    # Display conversation history in an expander
    with st.expander("**Conversation History**", expanded=True):
        for message in st.session_state['conversation_history']:
            st.markdown(message, unsafe_allow_html=True)

    st.sidebar.markdown("----")

    # Clear conversation history button
    if st.sidebar.button("Clear History"):
        st.session_state['conversation_history'].clear()


def save_audio_file(audio_bytes, file_extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    with open('./temp/'+ file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


def transcribe_audio(file_path):
    transcript = None
    try:
        with open(file_path, "rb") as audio_file:
            transcript = transcribe(audio_file)

        os.remove(file_path)
    except Exception as exc:
        st.error("The recorded file is too short. Please record your question again!")

    return transcript

if __name__ == "__main__":
    init_app()
    main()
