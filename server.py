from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json, urllib.request, urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi"   # fastest is "phi", use "mistral" for balance and "llama3" for best quality


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
            print("❌ GET error:", e)
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        print("🔥 USING OLLAMA BACKEND")

        # parse request
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as e:
            print("❌ Invalid JSON:", e)
            self.send_response(400)
            self.end_headers()
            return

        messages = body.get("messages", [])

        # build prompt (optimized)
        prompt_parts = []
        for m in messages[-6:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            prompt_parts.append(f"{role}: {m.get('content', '')}")
        prompt_parts.append("Assistant: ")

        prompt = "\n".join(prompt_parts)

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 200,
                        "temperature": 0.7
                    }
                }).encode(),
                headers={"Content-Type": "application/json"}
            )

            res = urllib.request.urlopen(req, timeout=60)

            # streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            for line in res:
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode())
                    token = chunk.get("response", "")
                    if token:
                        self.wfile.write(token.encode())
                        self.wfile.flush()
                except:
                    continue

        except urllib.error.URLError:
            print("❌ Ollama not running")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Ollama not running. Run: ollama run phi")

        except Exception as e:
            print("❌ Server error:", e)
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


print("🚀 Server running at http://localhost:8000")
ThreadingHTTPServer(("", 8000), Handler).serve_forever()