from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error

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

        # Build prompt from last 6 messages for context
        prompt_parts = []
        for m in messages[-6:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            prompt_parts.append(f"{role}: {m.get('content', '')}")

        prompt_parts.append(
            "Assistant: Give a clear, detailed answer in 4-6 sentences. "
            "Explain the key points thoroughly. Do not cut off mid-sentence."
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
                        "num_predict": 600,       # Raised to allow detailed multi-sentence answers
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.2,
                        "stop": ["\nUser:", "\nAssistant:", "```"]  # Added ``` to catch markdown artifacts
                    }
                }).encode(),
                headers={"Content-Type": "application/json"}
            )

            res = urllib.request.urlopen(req, timeout=60)

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            for line in res:
                if not line:
                    continue

                try:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")

                    if token:
                        # Write tokens directly — Ollama streams complete tokens,
                        # no need to buffer by word boundaries
                        self.wfile.write(token.encode("utf-8"))
                        self.wfile.flush()

                    # Break cleanly when Ollama signals it's done
                    if chunk.get("done"):
                        break

                except Exception:
                    continue

        except urllib.error.URLError:
            print("Ollama not running")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Ollama not running. Run: ollama run mistral")

        except Exception as e:
            print("Server error:", e)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Server error: {e}".encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Server running at http://localhost:8000  (model: {MODEL})")
    ThreadingHTTPServer(("", 8000), Handler).serve_forever()