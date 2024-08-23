from dotenv import load_dotenv
import os

from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_input = input("What is your question: ")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are an educational helper or teaching assistant. do not provide answers for any questions asked but you are able to help walk through steps "
        },
        {
            "role": "user",
            "content": chat_input,
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)