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

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

generation_config = GenerationConfig(
    do_stream=True,
    min_length=1,
    max_length=200,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
    length_penalty=1.0
)

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    if not data or "messages" not in data:
        return jsonify({"error": "No messages provided"}), 400

    messages = data["messages"]
    if not isinstance(messages, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
        return jsonify({"error": "Invalid message format"}), 400

    def stream():
        try:
            for chunk in pipe(messages, generation_config=generation_config, return_full_text=False):
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
