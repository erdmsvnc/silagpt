from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ API KEY BURADAN OKUNUYOR
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MEMORY_FILE = "memory.json"


class ChatRequest(BaseModel):
    message: str


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(role, content):
    memory = load_memory()
    memory.append({"role": role, "content": content})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message

    memory = load_memory()

    messages = [
        {
            "role": "system",
            "content": (
                "Sen SılaGpt'sin. Kullanıcıyı hatırlarsın. "
                "Sıcak, samimi ve destekleyici cevaplar verirsin."
            )
        }
    ]

    messages.extend(memory)
    messages.append({"role": "user", "content": user_message})

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7
    )

    reply = completion.choices[0].message.content

    save_memory("user", user_message)
    save_memory("assistant", reply)

    return {"reply": reply}
