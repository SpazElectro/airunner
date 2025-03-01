import os, requests
from time import time
from typing import List, TypedDict, Literal
from dotenv import load_dotenv
load_dotenv()

Role = Literal["user", "assistant", "system"]
class Message(TypedDict):
    role: Role
    content: str
def make_message(role: Role, content: str) -> Message:
    return { "role": role, "content": content }

URL = os.getenv("SERVER_URL")

def request(messages: List[Message]):
    print(messages)
    s = time()
    result = requests.post(f"{URL}/generate", json={"messages": messages})
    print(f"took {time()-s}s")
    if result.ok:
        print("request is ok!")
    print(result.text)

request([
    make_message("system", """
                Your name is "Jeff" and you like playing baseball.
                And you love (and mention a lot) the NBA
                """.replace("\n", "")),
    make_message("user", "hi! how are you?")
])