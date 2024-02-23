from langchain_community.tools import BaseTool
from langchain_community.llms import Clarifai as Clarifaillm
from langchain_community.vectorstores import Clarifai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import streamlit as st

class AcademiaKBTool(BaseTool):
    name = "Academia Knowledge Base"
    description = """Use this tool when the user prompt has inquiries about academic papers, literatures or research points. 
    Use also this tool to answer general questions or inquiries if the relevant answers exist in the knowledge base"""

    def __init__(self):
        super().__init__()

    def _run(self, prompt):
         # Initialize the prompt template
        prompt_template = PromptTemplate(
            template="""You are an expert academic researcher specialized in helping researchers to find answers and relavent
                        papers or sources for their domain or research. Use also this tool to answer general questions 
                        or inquiries if the relevant answers exist in the knowledge base. You should answer the questions 
                        based on the given context:
                        {context}
                        Question: {question}
                        Answer: """,
            input_variables=["context", "question"])

        # Initialize Clarifai vector store
        clarifai_vector_db = Clarifai(
            user_id=st.secrets["USER_ID"],
            app_id=st.secrets["APP_ID"],
            pat=os.getenv("CLARIFAI_PAT")
        )

        # Initialize RetrievalQA
        model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"

        llm = Clarifaillm(model_url=model_url)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                              retriever=clarifai_vector_db.as_retriever(),
                                              chain_type_kwargs={"prompt": prompt_template})
        return qa.run(prompt)
