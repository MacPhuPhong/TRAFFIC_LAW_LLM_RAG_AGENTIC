import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

try:
    result = genai.embed_content(
        model='models/embedding-001',
        content="hello",
        task_type='SEMANTIC_SIMILARITY'
    )
    print("Success: models/embedding-001 works")
except Exception as e:
    print("Error:", e)

