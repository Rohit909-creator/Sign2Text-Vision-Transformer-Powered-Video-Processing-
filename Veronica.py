"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""


# Shield Commands arent implemented now yet

from typing import Any
import google.generativeai as genai
import re
from pathlib import Path
import time

genai.configure(api_key="AIzaSyDxwzSTS5XWJVDsKe-luOk2Wuwe5CKTyvQ")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


prompt_parts = [
  """"You are a feature in a project called LLM Assist, which can functions as an autocorrect,\nyour job is to  make meaningful outputs based on the crippled inputs\n\nExample: \n\ninput: hllohwareu\noutput: hello  how are you\n\ninput: a apple fall t r e e\noutput: an apple fell from tree\n\ninput: how todotis thinig\noutput: how to do this thing\n\ninput: he go canteen in Hyderabad\noutput: he went to a canteen in Hyderabad\n\ninput: he called discuss bill from canteen\noutput: he called to discuss the bill from canteen\n\n"""",
]

class Chat_V:

    def __init__(self) -> None:

        self.model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
        
        self.vision = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
        
        self.response = None
        self.prompt_parts = prompt_parts

        self.image_parts = image_parts = []
        self.image_path = None
        
        self.count = 0
    def __call__(self, message):

        if self.image_path:
            # print("Inside Vision chat")
            message = f"Human: {message}"

            self.image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": Path(self.image_path).read_bytes()
                },
            ]

            # print("Image Content", Path(self.image_path).read_bytes())

            self.prompt_parts.append(self.image_parts[0])
            self.prompt_parts.append(message)
            self.response = self.vision.generate_content(self.prompt_parts)
            self.prompt_parts.append(f"Veronica: {self.response.text}")
            # self.image_path = None
            return self.response.text

        else:
            message = f"input: {message}"
            self.count += 1
            if self.count > 2:
                self.count = 0
                self.prompt_parts = prompt_parts

            self.prompt_parts.append(message)
            self.response = model.generate_content(self.prompt_parts)
            self.prompt_parts.append(f"Veronica: {self.response.text}")
            return self.response.text
        # print(response.text)
        
    def __next__(self):
        
        pattern = re.compile(r'\b\w+\(\)')

        # Search for the pattern in the message
        match = pattern.search(self.response.text)

        # If a match is found, print the matched substring
        if match:
            print("Extracted function call:", match.group())
            return match.group()
        else:
            print("No function call found in the message")
            return None
        
if __name__ == "__main__":

    chat = Chat_V()
    # for i in range(100):
    response = chat("1plus3*2")
    print(response)