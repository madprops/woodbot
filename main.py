import requests
import websocket
import json
import re
import traceback
import os
from datetime import datetime
from pathlib import Path
import sys
from llama_cpp import Llama
import threading
import time

HERE = Path(__file__).parent
username = os.environ.get("WOODY_USERNAME")
password = os.environ.get("WOODY_PASSWORD")

headers = {
    "User-Agent": "woody",
    "Origin": "https://deek.chat",
    "DNT": "1",
}

url = "https://deek.chat"
ws_url = "wss://deek.chat/ws"
prefix = ","
token = None
session = None
delay = 3

# model = "/media/storage3/models/tinyllama-1.1b-chat-v0.3.Q6_K.gguf"
model = "/media/storage3/models/pirouette-7b.Q5_K_M.gguf"
llama = None
streaming = False
system = f"Your name is woody and you respond to questions. Respond in 280 characters or less."


def msg(message: str) -> None:
    print(message, file=sys.stderr)


def auth():
    global token, session, headers

    if not username or not password:
        msg("Missing environment variables")
        exit(1)

    data = {"name": username, "password": password, "submit": "log+in"}
    res = requests.post(url + "/login/submit", headers=headers, data=data, allow_redirects=False)
    token = re.search("(?:api_token)=[^;]+", res.headers.get("Set-Cookie")).group(0)
    session = re.search("(?:session_id)=[^;]+", res.headers.get("Set-Cookie")).group(0)
    headers["Cookie"] = token + "; " + session


def run():
    ws = websocket.WebSocketApp(ws_url,
                                header=headers,
                                on_message=on_message)

    stop_event = threading.Event()

    def _run():
        while not stop_event.is_set():
            ws.run_forever()
            time.sleep(1)  # sleep for a bit to prevent busy-waiting

    wst = threading.Thread(target=_run)
    wst.start()

    try:
        while True:
            time.sleep(1)  # sleep main thread to prevent busy-waiting
    except KeyboardInterrupt:
        stop_event.set()  # signal the thread to stop
        ws.close()
        exit(0)


def on_message(ws, message):
    try:
        data = json.loads(message)
    except BaseException:
        return

    if data["type"] == "message":
        if streaming:
            return

        uname = data["data"]["name"]

        if uname == username:
            return

        text = data["data"]["text"].strip()

        if not text:
            return

        words = text.lstrip(prefix).split(" ")
        cmd = words[0]
        args = words[1:]
        argument = " ".join(args).strip()
        room_id = data["roomId"]

        if cmd == "!ai":
            respond(ws, room_id, argument, uname)


def respond(ws, room_id, text, uname):
    def wrapper(ws, room_id, text, uname) -> None:
        global streaming
        streaming = True
        stream(ws, room_id, text, uname)
        streaming = False

    thread = threading.Thread(target=wrapper, args=(ws, room_id, text, uname))
    thread.daemon = True
    thread.start()


def stream(ws, room_id, text, uname):
    print("Responding:", text, room_id, uname)
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": text[:140]})
    ws.send(json.dumps({"type": "messageEnd", "data": "Thinking...", "roomId": room_id}))
    ws.send(json.dumps({"type": "messageStart", "data": "Thinking...", "roomId": room_id}))

    try:
        output = llama.create_chat_completion_openai_v1(
            stream=True,
            max_tokens=360,
            messages=messages,
            seed=326,
        )
    except BaseException as e:
        print(e)
        return

    token_printed = False
    last_token = " "
    tokens = []
    last_date = 0.0

    def get_message():
        return "".join(tokens)

    def print_buffer() -> None:
        nonlocal last_date

        if not len(tokens):
            return

        datenow = time.time()
        message = get_message()

        if not message:
            return

        if (datenow - last_date) < 0.1:
            return

        last_date = datenow
        ws.send(json.dumps({"type": "messageChange", "data": message, "roomId": room_id}))

    for chunk in output:
        delta = chunk.choices[0].delta

        if hasattr(delta, "content"):
            token = delta.content

            if token == "\n":
                if not token_printed:
                    continue
            elif token == " ":
                if last_token == " ":
                    continue

            last_token = token

            if token is not None:
                if not token_printed:
                    token = token.lstrip()
                    token_printed = True

                tokens.append(token)
                print_buffer()

    ws.send(json.dumps({"type": "messageEnd", "data": get_message(), "roomId": room_id}))



while True:
    llama = Llama(
        model_path=model,
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=33,
        verbose=False,
        chat_format="chatml",
    )

    try:
        auth()
        msg("Authenticated")
        run()
    except KeyboardInterrupt:
        break
    except Exception as e:
        msg("(Main) Error:", e)
        traceback.print_exc()
