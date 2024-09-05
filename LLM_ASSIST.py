"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import os
import google.generativeai as genai


class Chat:

    def __init__(self):

        genai.configure(api_key="<api_key>")

        # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        system_instruction="You are a feature in a project called LLM Assist, which can functions as an autocorrect,\nyour job is to  make meaningful outputs based on the crippled inputs\n\nExample: \n\ninput: hllohwareu\noutput: hello  how are you\n\ninput: a apple fall t r e e\noutput: an apple fell from tree\n\ninput: how todotis thinig\noutput: how to do this thing\n\ninput: he go canteen in Hyderabad\noutput: he went to a canteen in Hyderabad\n\ninput: he called discuss bill from canteen\noutput: he called to discuss the bill from canteen\n\ninput: aeroplane (sign 1)1.png app1.png alarm sign 21.png call1.png\noutput: The aeroplanes alarm app started ringing",
        )

    def __call__(self, message:str):

        chat_session = self.model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                f"input: {message}",
            ],
            },
        ]
        )

        response = chat_session.send_message("INSERT_INPUT_HERE")

        # print(response.text)
        resp = response.text
        return resp[resp.index('output: ')+len("output: "):]

if __name__ == "__main__":

    chat = Chat()
    response = chat('he debit card from cashier')
    print(response)
