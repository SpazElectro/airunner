import json
import os
import threading
import subprocess
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import pipeline, GenerationConfig
from pyngrok import ngrok

ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
url = os.getenv("SERVER_URL")

if ngrok_auth_token:
    os.system(f"ngrok config add-authtoken {ngrok_auth_token}")
else:
    raise ValueError("ERROR: No ngrok auth token found in userdata!")

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B")

generation_config = GenerationConfig(
    do_stream=True,
    min_length=1,
    max_new_tokens=200,         # Control the number of new tokens generated
    temperature=0.7,            # For more coherent but creative output
    top_k=50,                   # Controls randomness
    top_p=0.9,                  # Nucleus sampling for better coherence
    repetition_penalty=1.2,     # Penalize repetition
    no_repeat_ngram_size=2,     # Avoid repeating phrases
    length_penalty=1.0,         # Balanced output length
    do_sample=True              # Enable sampling with temperature/top_p
)

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    if not data or "messages" not in data:
        return jsonify({"error": "No messages provided"}), 400

    messages = data["messages"]
    if not isinstance(messages, list) or not all(
        isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages
    ):
        return jsonify({"error": "Invalid message format"}), 400

    prompt_lines = []
    for msg in messages:
        if msg["role"] == "user":
            prompt_lines.append("User: " + msg["content"])
        elif msg["role"] == "assistant":
            prompt_lines.append("Assistant: " + msg["content"])
    prompt = "\n".join(prompt_lines) + "\nAssistant: "

    def stream():
        try:
            for chunk in pipe(prompt, generation_config=generation_config, return_full_text=False):
                yield json.dumps({"response": chunk["generated_text"]}) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"An error occurred: {str(e)}"}) + "\n"

    return Response(stream(), content_type="application/json")

def run_ngrok():
    process = subprocess.Popen(
        ["ngrok", "http", "--url=" + url, "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

ngrok_thread = threading.Thread(target=run_ngrok, daemon=True)
ngrok_thread.start()

app.run(host="0.0.0.0", port=5000)
