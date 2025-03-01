import json, os, threading, subprocess
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import pipeline, GenerationConfig

ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
url = os.getenv("SEVER_URL")

if ngrok_auth_token:
    os.system(f"ngrok config add-authtoken {ngrok_auth_token}")
else:
    raise ValueError("ERROR: No ngrok auth token found in userdata!")

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

generation_config = GenerationConfig(
    do_stream=True,
    max_new_tokens=100
)

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    messages = data.get("messages", [{"role": "user", "content": "Hello!"}])

    def stream():
        for chunk in pipe(messages, generation_config=generation_config, return_full_text=False):
            yield json.dumps({"response": chunk["generated_text"]}) + "\n"

    return Response(stream(), content_type="application/json")

def run_ngrok():
    process = subprocess.Popen(
        ["ngrok", "http", "--url="+url, "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

ngrok_thread = threading.Thread(target=run_ngrok, daemon=True)
ngrok_thread.start()

app.run(host="0.0.0.0", port=5000)
