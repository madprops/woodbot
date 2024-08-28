import requests
import websocket
import json
import re
import traceback
import os
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
ws = None


# LLAMA CONFIG

# model = "/media/storage3/models/tinyllama-1.1b-chat-v0.3.Q6_K.gguf"
# model = "/media/storage3/models/pirouette-7b.Q5_K_M.gguf"
model = "/media/storage1/models/llama3.gguf"
default_system = f"You can talk about anything. You are a bit rude and unhinged. You make subtle bird references sometimes. Respond in 280 characters or less."
system = default_system
context = 2048
chat_format = "chatml"
threads = 6
gpu_layers = 33
llama = None
streaming = False
verbose = False
max_input = 200
stop = ["<|im_start|>", "<|im_end|>"]
history = []
default_history = 0
used_history = default_history
max_history = 100

#############


def msg(*args) -> None:
    print(*args, file=sys.stderr)


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
    global ws

    try:
        if ws and ws.sock and ws.sock.connected:
            return
    except NameError:
        pass

    ws = websocket.WebSocketApp(ws_url,
                                header=headers,
                                on_message=on_message)

    stop_event = threading.Event()

    def _run():
        ws.run_forever()

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
    global history, used_history, system

    try:
        data = json.loads(message)
    except BaseException:
        return

    if data["type"] in ["message", "messageEnd"]:
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

        if cmd in ["!ai", "!a", "!i", ".ai", ";ai", ",ai",
                   "woody,", "woddy,", "woody:", "woddy:", "wody,", "wody:",
                   "wood,", "wood:", "woffy,", "woffy:", "woofy,", "woofy:"]:

            if argument == "help":
                help_msgs = [
                    "clear",
                    "history = [ int | default ]",
                    "system = [ str | default ]",
                ]

                help_msg = f"Commands: {', '.join(help_msgs)}"
                send_message(ws, help_msg, room_id)
            elif argument == "clear":
                history = []
                send_message(ws, "History cleared", room_id)
            elif argument == "history":
                send_message(ws, f"History: {used_history}", room_id)
            elif argument.startswith("history = "):
                new_history = argument.replace("history = ", "", 1).strip()

                if new_history == "default":
                    used_history = default_history
                    send_message(ws, f"History set to default", room_id)
                elif new_history and new_history.isdigit():
                    new_history = int(new_history)

                    if new_history >= 0 and new_history <= max_history:
                        used_history = int(new_history)
                        send_message(ws, f"History set to {used_history}", room_id)
            elif argument == "system":
                send_message(ws, f"System: {system}.", room_id)
            elif argument.startswith("system = "):
                new_system = argument.replace("system = ", "", 1).strip()

                if new_system:
                    if new_system == "default":
                        system = default_system
                        send_message(ws, f"System set to default", room_id)
                    else:
                        system = new_system
                        send_message(ws, f"System prompt changed", room_id)
            else:
                respond(ws, room_id, argument, uname)


def send_message(ws, text, room_id):
    ws.send(json.dumps({"type": "message", "data": text, "roomId": room_id}))


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
    global history

    def send_message(what: str, message: str) -> None:
        ws.send(json.dumps({"type": what, "data": message, "roomId": room_id}))

    msg(f"Responding:", text, room_id, uname)
    messages = [{"role": "system", "content": system}]

    if history and used_history:
        messages.extend(history[-used_history :])

    first_message = {"role": "user", "content": text[:max_input]}
    history.append(first_message)
    messages.append(first_message)

    send_message("messageEnd", "Thinking...")
    send_message("messageStart", "Thinking...")

    try:
        output = llama.create_chat_completion_openai_v1(
            stream=True,
            max_tokens=360,
            messages=messages,
            stop=stop,
        )
    except BaseException as e:
        msg(e)
        return

    token_printed = False
    last_token = " "
    tokens = []
    last_date = 0.0

    def get_message():
        return "".join(tokens)

    def send_tokens() -> None:
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
        send_message("messageChange", message)

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
                send_tokens()

    response = get_message()
    history.append({"role": "assistant", "content": response})

    if len(history) > max_history:
        history = history[-max_history :]

    send_message("messageEnd", response)

llama = Llama(
    model_path=model,
    n_ctx=context,
    n_threads=threads,
    n_gpu_layers=gpu_layers,
    verbose=verbose,
    chat_format=chat_format,
)

while True:
    try:
        auth()
        msg("Authenticated")
        run()
    except KeyboardInterrupt:
        break
    except Exception as e:
        msg(f"(Main) Error:", e)
        traceback.print_exc()
