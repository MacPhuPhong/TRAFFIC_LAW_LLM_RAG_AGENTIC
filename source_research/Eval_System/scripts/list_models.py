import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)
