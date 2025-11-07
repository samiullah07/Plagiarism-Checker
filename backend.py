from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List

load_dotenv()

client = Groq()


def finger_print_data(text):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are an AI text analyst. Analyze the input text and estimate its percentage uniqueness "
            "as a fingerprint (0-100), compared to general online sources. "
            "If the fingerprint is more than 80% unique, respond 'Unique: True'. "
            "Otherwise, respond 'Unique: False'. "
            "At the top of your answer, ONLY output either 'Unique: True' or 'Unique: False'."
            "Then give a short explanation and the estimated uniqueness percentage.\n\n"
            "Text:\n{text}"
        ),
    )
    rendered = prompt.format(text=text)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": rendered}
        ],
    )

    try:
        content = response.choices[0].message.content
        # Check for "Unique: True" or "Unique: False" in the response
        if content is None:
            is_unique = None  # No content returned
        else:
            text = content.strip()
            if text.startswith("Unique: True"):
                is_unique = True
            elif text.startswith("Unique: False"):
                is_unique = False
            else:
                is_unique = None  # Unclear response
        return {
            "ai_result": content,
            "is_unique": is_unique
        }
    except Exception:
        return {"ai_result": str(response), "is_unique": None}
