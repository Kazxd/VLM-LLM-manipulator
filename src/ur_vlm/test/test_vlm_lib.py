"""Run: python3 src/ur_vlm/test/test_vlm_lib.py   (needs Pillow, no ROS, no Ollama)"""
import json, math, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
from ur_vlm import vlm_lib as V

F = 160.0 / math.tan(1.05 / 2)                   # 320x240 camera, 1.05 rad horizontal fov
K = [F, 0, 160.0, 0, F, 120.0, 0, 0, 1]

# projection: the exact inverse of detect_objects.py (x = 0.6 - (v-cy)*d/fy, y = -(u-cx)*d/fx)
for x, y, top in ((0.5, 0.1, 0.25), (0.7, 0.12, 0.25), (0.6, -0.15, 0.25), (0.38, 0.34, 0.25)):
    u, v = V.project(x, y, top, K)
    d = V.CAM_POS[2] - top
    assert abs((0.6 - (v - K[5]) * d / K[4]) - x) < 1e-9 and abs((-(u - K[2]) * d / K[0]) - y) < 1e-9
u_l, _ = V.project(0.6, 0.2, 0.25, K); u_r, _ = V.project(0.6, -0.2, 0.25, K)
assert u_l < u_r                                  # robot's left (+y) is image left
_, v_far = V.project(0.8, 0.0, 0.25, K); _, v_near = V.project(0.4, 0.0, 0.25, K)
assert v_far < v_near                             # far from the robot is image top

# annotate
img = V.annotate(Image.new("RGB", (320, 240), (90, 70, 40)), {"red": (100, 100), "can": (10, 5)})
assert img.size == (960, 720)

# schema / parse
names = ["red", "blue", "can"]
assert V.schema(names)["properties"]["objects"]["items"]["enum"] == names
p = V.parse_answer('{"answer":"blue","objects":["Blue","purple","blue"],"ambiguous":false,"clarification":""}', names)
assert p["ok"] and p["objects"] == ["blue"] and p["dropped"] == ["purple"], p
p = V.parse_answer('```json\n{"answer":"?","objects":[],"ambiguous":true,"clarification":""}\n```', names)
assert p["ok"] and p["ambiguous"] and p["clarification"] == "Which object do you mean?", p
p = V.parse_answer('<think>x</think>{"answer":"a","objects":["red"],"ambiguous":false,"clarification":"ignored"}', names)
assert p["ok"] and p["clarification"] == "", p
assert not V.parse_answer("not json", names)["ok"]
assert not V.parse_answer(None, names)["ok"]

# a scripted Ollama server: checks the request (image, enum schema, no thinking) and the reply handling
class H(BaseHTTPRequestHandler):
    last = None
    def do_POST(self):
        H.last = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"message": {"role": "assistant", "content":
                           '{"answer":"The blue cube is closest.","objects":["blue","orange"],"ambiguous":false,"clarification":""}'},
                           "total_duration": 2_000_000_000}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
out = V.ask(f"http://127.0.0.1:{srv.server_port}", "qwen3-vl:4b", "which is closest to the can?", img, names)
r = H.last
assert r["model"] == "qwen3-vl:4b" and r["think"] is False and r["keep_alive"] == "0"
assert r["format"]["properties"]["objects"]["items"]["enum"] == names
assert len(r["messages"][1]["images"]) == 1 and "closest to the can" in r["messages"][1]["content"]
assert out["ok"] and out["objects"] == ["blue"] and out["dropped"] == ["orange"] and abs(out["total_s"] - 2.0) < 1e-9
print("all vlm_lib tests passed")
