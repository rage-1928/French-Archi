from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"   # "phi" (fast), "mistral" (balanced), "llama3" (best quality)

SYSTEM_PROMPT = (
    "You are a direct, factual assistant. Rules you must follow without exception:\n"
    "- Answer immediately with real facts. Never restate or summarise the question.\n"
    "- Never ask clarifying questions or follow-up questions.\n"
    "- Never list steps, sub-tasks, or say what you are about to do.\n"
    "- Write in complete sentences. Never stop mid-sentence.\n"
    "- Be specific and concise. One coherent paragraph is ideal.\n"
    "- NEVER use hypothetical scenarios, invented examples, or 'imagine if' framings.\n"
    "- NEVER say 'for example', 'suppose', 'consider a case where', 'let's say', or similar.\n"
    "- NEVER invent people, companies, or situations to illustrate a point.\n"
    "- If a real example is needed, use a documented real-world fact — otherwise skip it.\n"
    "- NEVER describe or announce what your response will do. Never say things like "
    "'This response provides...', 'I will now explain...', 'As required by the rules...', "
    "or any similar meta-commentary about your own reply. Just answer.\n"
)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        try:
            # Map URL path to a local HTML file; default to index.html
            path = self.path.split("?")[0]  # strip query strings
            if path in ("/quiz.html", "/quiz"):
                filename = "quiz.html"
            else:
                filename = "index.html"

            with open(filename, "rb") as f:
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

        # System prompt goes first — before any conversation turns
        prompt_parts = [SYSTEM_PROMPT]

        # Build conversation history from last 6 messages
        for m in messages[-6:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            prompt_parts.append(f"{role}: {m.get('content', '')}")

        # End with a clean "Assistant:" cue — no instructions here
        prompt_parts.append("Assistant:")

        prompt = "\n".join(prompt_parts)

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 800,       # Higher ceiling to avoid mid-sentence cutoff
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.2,
                        "stop": ["\nUser:", "\nAssistant:"]  # Removed ``` — it was cutting responses short
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
                        self.wfile.write(token.encode("utf-8"))
                        self.wfile.flush()

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