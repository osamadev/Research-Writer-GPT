import os
import re
import datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.agents import initialize_agent, AgentType, load_tools, create_react_agent, AgentExecutor
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun, SemanticScholarAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from helpers.images_analyzer_tool import ImageAnalyzerFromURL, ImageAnalyzerFromFile
from langchain_community.chat_models import ChatOpenAI
from audio_recorder_streamlit import audio_recorder
from langchain.schema.messages import SystemMessage
from langchain_community.tools import PubmedQueryRun
from helpers.academia_kb_tool import AcademiaKBTool
from helpers.CustomStreamingStdOutCallbackHandler import CustomStreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from typing import List
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import StringPromptTemplate
import base64
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )
from langchain.prompts import PromptTemplate
from helpers.presentation_generator import *
from streamlit_extras.switch_page_button import switch_page

def transcribe(audio_file):
    from openai import OpenAI
    client = OpenAI()
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    return transcript

# Load environment variables
# load_dotenv()

def init_app():
    os.environ["CLARIFAI_PAT"] = st.secrets["CLARIFAI_PAT"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["SERPAPI_API_KEY"] = st.secrets["SERP_API_KEY"]

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
    from openai import OpenAI
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    return response.content

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

def build_prompts():
    with open('prompts/main.txt', 'r') as file:
        content = file.read()

    prompt: PromptTemplate = PromptTemplate(
        input_variables=["input"],
        template=content)
    return prompt

# Initialize the Langchain agent
def initialize_chatbot():

    openai_model = "gpt-4-0125-preview"

    llm = ChatOpenAI(model_name=openai_model, temperature=0.2 ,streaming=True)

    tools = load_tools()

    chat_agent = create_react_agent(
        llm=llm, tools=tools, prompt=build_prompts())

    agent_executor = AgentExecutor(agent=chat_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor

def load_tools():
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

    image_analyzer_tool = ImageAnalyzerFromURL()
    search = SerpAPIWrapper()

    repl_tool = Tool(
    name="google_search",
    description="Use this tool to search for more details to enrich the final response, and to provide grounded resources and references for the papers and articles.",
    func=search.run)
    
    tools = [semantic_scholar_tool, google_scholar_tool, repl_tool, pubmed_research_tool, arxiv_tool, image_analyzer_tool]
    return tools


def generate_speech(text):
    try:
        return text_to_speech(text)
    except Exception as e:
        st.error(f"Error in text-to-speech generation: {e}")
        return None

def process_user_input(user_input, chat_agent):
    conversation_history = st.session_state['conversation_history']
    user_prompt = f"""You are an expert researcher who can help other researchers from different domains to 
        find relevant resources and papers for their research domain. You should answer the questions 
        based on the given context:
        {conversation_history}
        Question: {user_input}
        Answer: """
    # Update conversation history with user query
    conversation_history.append({'type': 'human', 'content': f"👤 You: {user_input}"})
    st.session_state['user_prompts'].append(user_input)
    st.session_state['conversation_history'] = conversation_history
    
    # Get response from llm
    response = chat_agent.invoke({'input': user_prompt})
    if 'output' in response:
        new_response = f"🤖 AI: {response['output']}"
        conversation_history.append({'type': 'AI', 'content': new_response})
        st.session_state['conversation_history'] = conversation_history

def main():
    st.set_page_config("Research Assistant GPT", layout="wide", page_icon="🎓")
    st.title("🎓 Research Assistant GPT 🤖")
    st.caption("Explore the world of academia with ease 🎓! Our AI-powered research assistant 🤖 is here to help you discover and recommend scholarly papers 📚, provide insightful summaries 📄, and guide you through a sea of literature with style and efficiency 🌟.")
        # Initialize chatbot
    chat_agent = initialize_chatbot()

    # Initialize session state for conversation history and audio transcription setting
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'user_prompts' not in st.session_state:
        st.session_state['user_prompts'] = []
    
    checkbox_key = 'enable_audio_transcription'

    # Initialize the session state for the checkbox if it hasn't been initialized
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = False

    st.checkbox("Enable audio transcription of responses", value=st.session_state[checkbox_key], key=checkbox_key)

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
        if st.session_state['uploaded_image']:
            # Update conversation history with user query
            conversation_history = st.session_state['conversation_history']
            conversation_history.append({'type': 'human', 'content': f"👤 You: {user_input}"})
            st.session_state['user_prompts'].append(user_input)
            st.session_state['conversation_history'] = conversation_history
            
            image_analyzer = ImageAnalyzerFromFile()
            tool_input = {
                "base64_image": base64.b64encode(st.session_state['uploaded_image'].getvalue()).decode(),
                "prompt": user_input
            }
            response = image_analyzer.run(tool_input, callbacks=[CustomStreamingStdOutCallbackHandler()])
            if response:
                new_response = f"🤖 AI: {response}"
                st.session_state['conversation_history'].append({'type': 'text', 'content': new_response})
        else:
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

        conversation_history = st.session_state['conversation_history']
        user_prompt = f"""You are an expert researcher who can help other researchers from different domains to 
        find relevant resources and papers for their research domain. You should answer the questions 
        based on the given context:

        Question: {transcript_text}
        Answer: """

        conversation_history = st.session_state['conversation_history']
        conversation_history.append({'type': 'human', 'content': f"👤 You: {transcript_text}"})
        st.session_state['conversation_history'] = conversation_history

        # Get response from chatbot
        response = chat_agent.invoke({'input': user_prompt})

        if 'output' in response:
            new_response = f"🤖 AI: {response['output']}"
            st.session_state['conversation_history'].append({'type': 'AI', 'content': new_response})

        # Clear the recorded audio to allow new recording
        audio_bytes = None

    # Display conversation history in an expander
    with st.expander("**Conversation History**", expanded=True):
        for message in st.session_state['conversation_history']:
            st.markdown(message['content'], unsafe_allow_html=True)
            # Transcribe response to audio if enabled
            if st.session_state.get(checkbox_key, True) and message['type'] == "AI":
                try:
                    speech_audio = text_to_speech(message['content'])
                    st.audio(speech_audio, format='audio/mp3', start_time=0)
                except Exception as e:
                    st.error(f"Failed to convert text to speech: {e}")

    with st.sidebar:
        st.markdown("## Research Assistant GPT")
        
        col1, col2 = st.columns([1,1])
        # Button to generate PowerPoint slides
        if col1.button('Generate Slides'):
            if 'conversation_history' in st.session_state:
                conversation_history = " ".join([item['content'] for item in st.session_state['conversation_history']])
                
                ppt_file, generated_content = presentation_tool.invoke(conversation_history)
                
                # Ensure ppt_file is not None or empty
                if ppt_file.getbuffer().nbytes == 0:
                    st.error("The generated PowerPoint file is empty.")
                else:
                    st.download_button(
                        label="Download Slides",
                        data=ppt_file,
                        file_name="conversation_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    slides = extract_presentation_content(generated_content)
                    expander_title = "**📑 Preview Slides Content**"  
                    with st.sidebar.expander(expander_title, expanded=True):
                        for slide in slides:
                            st.subheader(f"{slide.get('title').strip()}")
                            st.write(slide.get('content').strip())
                            st.markdown("---")
            else:
                st.sidebar.info("No conversation history to generate presentation.")

        # Clear conversation history button
        if col2.button("Clear History"):
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
        st.error(exc)
        st.error("The recorded file is too short. Please record your question again!")

    return transcript

if __name__ == "__main__":

    if "authentication_status" not in st.session_state \
        or st.session_state["authentication_status"] == None or st.session_state["authentication_status"] == False:
        switch_page("Home")
        
    init_app()
    if st.session_state["authentication_status"]:
        main()
