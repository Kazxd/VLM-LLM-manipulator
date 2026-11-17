"""ROS-free VLM helpers (Pillow + stdlib only, unit-testable).

Idea (set-of-mark): the detectors give verified object positions. We draw each object's NAME on the overhead image at
its projected position and ask the VLM a question. The answer is constrained (JSON schema enum) to the names that were
actually detected, and is validated again here, so the VLM can interpret relations / ambiguity but cannot invent objects
or positions.

Image convention (overhead camera, see detect_objects.py): image top = far from the robot (+x), image bottom = near the
robot, image left = the robot's left (+y), image right = the robot's right (-y).
"""
import base64, io, json, re, urllib.request

CAM_POS = (0.6, 0.0, 1.2)            # must match the world file / detect_objects.py
NAMES = ("red", "green", "blue", "can", "ball", "block")
SCALE = 3                            # the 320x240 camera image is enlarged so labels are legible
MARK = {"red": (255, 60, 60), "green": (60, 220, 60), "blue": (80, 140, 255),
        "can": (255, 220, 0), "ball": (200, 100, 255), "block": (240, 240, 240)}


def project(x, y, top, k):
    """World (x, y) at surface height `top` -> pixel (u, v). Exact inverse of the mapping in detect_objects.py.
    k = CameraInfo.k (fx = k[0], cx = k[2], fy = k[4], cy = k[5])."""
    d = CAM_POS[2] - top
    return k[2] - (y - CAM_POS[1]) * k[0] / d, k[5] - (x - CAM_POS[0]) * k[4] / d


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                 # old Pillow
        return ImageFont.load_default()


def annotate(pil, marks, scale=SCALE):
    """marks = {name: (u, v)} in ORIGINAL pixels. Returns a new, enlarged RGB image with rings, name labels and
    orientation hints (FAR / NEAR ROBOT / LEFT / RIGHT)."""
    from PIL import Image, ImageDraw
    img = pil.convert("RGB").resize((pil.width * scale, pil.height * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(img)
    f, fs = _font(22), _font(16)
    for name, (u, v) in marks.items():
        cx, cy, r = u * scale, v * scale, 7 * scale
        col = MARK.get(name, (255, 255, 255))
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
        tw = dr.textlength(name, font=f)
        tx, ty = cx - tw / 2, cy - r - 28
        if ty < 2:
            ty = cy + r + 4
        dr.rectangle([tx - 3, ty - 1, tx + tw + 3, ty + 25], fill=(0, 0, 0))
        dr.text((tx, ty), name, fill=col, font=f)
    W, H = img.size
    for text, xy in (("FAR", (W // 2 - 14, 2)), ("NEAR ROBOT", (W // 2 - 50, H - 20)),
                     ("LEFT", (4, H // 2)), ("RIGHT", (W - 48, H // 2))):
        dr.text(xy, text, fill=(255, 255, 0), font=fs)
    return img


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def schema(names):
    return {"type": "object",
            "properties": {"answer": {"type": "string"},
                           "objects": {"type": "array", "items": {"type": "string", "enum": list(names)}},
                           "ambiguous": {"type": "boolean"},
                           "clarification": {"type": "string"}},
            "required": ["answer", "objects", "ambiguous", "clarification"]}


SYSTEM = """You look at an overhead photo of a robot table. Each object has its NAME written next to it; the names are \
exact and verified. Image top = far from the robot, bottom = near the robot, image left = robot's left, image right = \
robot's right. Answer only about the named objects.
Reply as JSON: answer (one short sentence), objects (the names the answer refers to, most relevant first), ambiguous \
(true if the question does not single out one answer, e.g. two equally close objects or an unknown description), \
clarification (a short question for the user if ambiguous, else an empty string).
Never invent an object that is not labelled."""


def build_messages(question, names, img_b64):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Labelled objects: {', '.join(names)}.\nQuestion: {question}",
             "images": [img_b64]}]


def parse_answer(text, names):
    """Validate the VLM reply against the detected names. Returns a dict, ok=False when it is not usable."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```\w*|```$", "", t.strip()).strip()
    try:
        d = json.loads(t)
        assert isinstance(d, dict)
    except (ValueError, AssertionError):
        return {"ok": False, "answer": t[:200], "objects": [], "ambiguous": False, "clarification": "", "dropped": []}
    objs, dropped = [], []
    for o in d.get("objects") or []:
        n = str(o).lower().strip()
        if n in names:
            if n not in objs:
                objs.append(n)
        else:
            dropped.append(str(o))
    amb = bool(d.get("ambiguous"))
    clar = str(d.get("clarification") or "").strip()
    if amb and not clar:
        clar = "Which object do you mean?"
    return {"ok": True, "answer": str(d.get("answer") or "").strip(), "objects": objs,
            "ambiguous": amb, "clarification": clar if amb else "", "dropped": dropped}


def ask(host, model, question, img, names, keep_alive="0", num_ctx=4096, num_predict=200, timeout=180):
    """One VLM call. `img` is the annotated PIL image. Returns parse_answer(...) plus raw text and timing."""
    body = json.dumps({"model": model, "messages": build_messages(question, names, png_b64(img)),
                       "stream": False, "think": False, "format": schema(names),
                       "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
                       "keep_alive": keep_alive}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    raw = data["message"].get("content", "")
    out = parse_answer(raw, names)
    out["raw"] = raw
    out["total_s"] = data.get("total_duration", 0) / 1e9
    return out
