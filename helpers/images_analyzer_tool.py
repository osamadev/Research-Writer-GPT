import re
from langchain.tools import BaseTool
from clarifai.client.model import Model
from clarifai.client.input import Inputs

class ClarifaiImageAnalyzerFromURL(BaseTool):
    name = "Image Captioning and Description"
    description = """A tool to analyze, describe and generate insights about the images or videos. 
    Use this tool when the user prompt includes a request to describe or analyse an image or video. 
    The image or video URL should be provided in the user prompt, otherwise do not use this tool.
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

        model = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision")
        inference_params = {'temperature': 0.2, 'max_tokens': 4000}
        clarifai_inputs = Inputs.get_multimodal_input(input_id="", image_url=image_url, raw_text=text_prompt)
        model_prediction = model.predict(inputs=[clarifai_inputs], inference_params=inference_params)
        return model_prediction.outputs[0].data.text.raw
    

class ClarifaiImageAnalyzerFromFile(BaseTool):
    name = "Image Analyzer Tool"
    description = """A tool to analyze, describe and generate insights about the images or videos. 
    Use this tool when the user prompt includes a request to describe or analyse an image or video. 
    The image or video URL should be provided in the user prompt, otherwise do not use this tool.
    Make sure to pass the whole user prompt in the 'action_input' parameter."""

    def __init__(self):
        super().__init__()

    def _run(self, **kwargs):
        file_bytes = kwargs.get('file_bytes')
        prompt = kwargs.get('prompt')
        # Prepare inference parameters
        inference_params = {'temperature': 0.2, 'max_tokens': 4000}

        # Perform model prediction
        model = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision")
        clarifai_inputs = Inputs.get_multimodal_input(input_id="", image_bytes=file_bytes, raw_text=prompt)
        model_prediction = model.predict(inputs=[clarifai_inputs], inference_params=inference_params)
        return model_prediction.outputs[0].data.text.raw