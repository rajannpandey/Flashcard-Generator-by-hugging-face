import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load variables from .env
load_dotenv()

# Get Hugging Face token
api_key = os.getenv("HF_TOKEN")

if not api_key:
    print("Error: HF_TOKEN not found in .env file.")
    exit()

# Create inference client
client = InferenceClient(
    provider="auto",
    api_key=api_key
)

# Take notes from user
print("===== AI FLASH CARD GENERATOR =====")

notes = input("\nEnter your topic or study notes:\n")

# Create prompt
prompt = f"""
You are a flashcard generator.

Create 5 simple study flashcards from the following notes.

Rules:
1. Generate exactly 5 flashcards.
2. Keep questions short and clear.
3. Keep answers simple.
4. Use only the information provided.
5. Use this exact format:

Flashcard 1
Q: Question
A: Answer

Flashcard 2
Q: Question
A: Answer

Notes:
{notes}
"""

try:
    # Send request to Hugging Face model
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    # Display generated flashcards
    print("\n===== GENERATED FLASHCARDS =====\n")
    print(response.choices[0].message.content)

except Exception as error:
    print("\nSomething went wrong:")
    print(error)