import re
import streamlit as st
from langchain_community.tools import BaseTool
from openai import OpenAI
import base64
import requests

class ImageAnalyzerFromURL(BaseTool):
    name = "Image Captioning and Description"
    description = """A tool to analyze, describe and generate insights about the images. 
    Use this tool when the user prompt includes a request to describe or analyse an image. 
    The image URL should be provided in the user prompt, otherwise do not use this tool.
    Make sure to pass the whole user prompt in the 'action_input' parameter."""

    def __init__(self):
        super().__init__()

    def _run(self, prompt):
        # Extracting URL from the prompt
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', prompt)
        if urls:
            image_url = urls[0]
            text_prompt = prompt.replace(image_url, '').strip()
        else:
            image_url = None
            text_prompt = prompt

        if text_prompt is None or text_prompt == '':
            text_prompt = "Describe and analyze this image or video."

        client = OpenAI()

        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{text_prompt}"},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{image_url}",
                    "detail": "high"
                },
                },
            ],
            }
        ],
        max_tokens=4000,
        )

        return response.choices[0].message.content
    

class ImageAnalyzerFromFile(BaseTool):
    name = "Image Analyzer Tool"
    description = """A tool to analyze, describe and generate insights about the images. 
    Use this tool when the user prompt includes a request to describe or analyse an image. 
    Use this tool if the image is uploaded and provided as base64 string."""

    def __init__(self):
        super().__init__()

    def _run(self, **kwargs):
        base64_image = kwargs.get('base64_image')
        prompt = kwargs.get('prompt')

        # OpenAI API Key
        api_key = st.secrets["OPENAI_API_KEY"]


        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 4000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')