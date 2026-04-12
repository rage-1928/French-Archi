from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"   # "phi" (fast), "mistral" (balanced), "llama3" (best quality)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        try:
            with open("index.html", "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            print("GET error:", e)
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as e:
            print("Invalid JSON:", e)
            self.send_response(400)
            self.end_headers()
            return

        messages = body.get("messages", [])

        # build prompt (controlled + concise)
        prompt_parts = []
        for m in messages[-6:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            prompt_parts.append(f"{role}: {m.get('content', '')}")

        prompt_parts.append(
            "Assistant: Answer in 2 concise sentences. "
            "Do not cut off mid-sentence. Be clear and precise."
        )

        prompt = "\n".join(prompt_parts)

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 160,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.2,
                        "stop": ["\nUser:", "\nAssistant:"]
                    }
                }).encode(),
                headers={"Content-Type": "application/json"}
            )

            res = urllib.request.urlopen(req, timeout=60)

            # streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            buffer = ""

            for line in res:
                if not line:
                    continue

                try:
                    chunk = json.loads(line.decode())
                    token = chunk.get("response", "")

                    if token:
                        buffer += token

                        # send only full words (avoid broken tokens)
                        if " " in buffer:
                            parts = buffer.rsplit(" ", 1)
                            safe_text = parts[0] + " "
                            buffer = parts[1]

                            self.wfile.write(safe_text.encode())
                            self.wfile.flush()

                except Exception:
                    continue

            # flush remaining buffer
            if buffer:
                self.wfile.write(buffer.encode())
                self.wfile.flush()

        except urllib.error.URLError:
            print("Ollama not running")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Ollama not running. Run: ollama run phi")

        except Exception as e:
            print("Server error:", e)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Server error")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Server running at http://localhost:8000")
    ThreadingHTTPServer(("", 8000), Handler).serve_forever()