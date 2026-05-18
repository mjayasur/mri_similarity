"""Static server for the triplet reader study + POST /submit -> responses.csv.

Serves this `website/` directory (which contains `index.html`, `triplets.json`,
and a `data/` symlink to the chosen DICOM dataset root). Run:

    cd website && python server.py            # http://127.0.0.1:8077

Expose publicly with `cloudflared tunnel --url http://127.0.0.1:8077` (or ngrok).
The DICOM data here is meant to be the public, de-identified LumbarDISC dataset;
the tunnel URL is unauthenticated, so do not point it at PHI.
"""
import os
import json
import csv
import datetime
import http.server
import socketserver

WEB = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("MRI_SIM_PORT", "8077"))
RESP = os.path.join(WEB, "responses.csv")
os.chdir(WEB)


class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".dcm"):
            return "application/dicom"
        return super().guess_type(path)

    def do_POST(self):
        if self.path.rstrip("/") == "/submit":
            n = int(self.headers.get("Content-Length", 0))
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                d = {}
            d.setdefault("server_ts", datetime.datetime.now().isoformat())
            new = not os.path.exists(RESP)
            with open(RESP, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["server_ts", "ts", "triplet_id", "choice",
                                "ref_study", "a_study", "b_study"])
                w.writerow([d.get("server_ts"), d.get("ts"), d.get("triplet_id"),
                            d.get("choice"), d.get("ref_study"),
                            d.get("a_study"), d.get("b_study")])
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
    print(f"serving {WEB} on http://127.0.0.1:{PORT}")
    httpd.serve_forever()
