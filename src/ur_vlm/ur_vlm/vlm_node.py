"""/query_scene service: verified detections + annotated overhead image -> Ollama VLM (qwen3-vl).

  ros2 run ur_vlm vlm_node
  ros2 run ur_vlm vlm_node --ros-args -p model:=qwen3-vl:4b -p keep_alive:=0
  ros2 service call /query_scene ur_interfaces/srv/QueryScene "{question: 'which object is closest to the can?'}"

Needs: sim (arm at home, out of the camera's view), detect_objects, and detect_open for can/ball/block.
keep_alive "0" unloads the VLM after every call so it does not fight qwen3:4b and Gazebo for the 6 GB of VRAM
(use "5m" if you have the memory and want fast repeated queries).
Annotated image of the last call: /tmp/vlm_last.png
"""
import math, threading, time
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ur_interfaces.srv import DetectObjects, DetectOpen, QueryScene
from ur_vlm import vlm_lib as V

DEBUG_IMAGE = "/tmp/vlm_last.png"
COLORS = ("red", "green", "blue")
# open-vocabulary labels: same wording as ur_skills/skill_server.py OBJECTS
EXTRA_LABELS = {"can": "yellow circle", "ball": "purple ball", "block": "white block"}
TABLE_TOP, CUBE, DUP_RADIUS = 0.20, 0.05, 0.04


class VlmNode(Node):
    def __init__(self):
        super().__init__("vlm_node", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.cb = ReentrantCallbackGroup()
        self.model = self.declare_parameter("model", "qwen3-vl:4b").value
        self.host = self.declare_parameter("host", "http://localhost:11434").value
        self.keep_alive = str(self.declare_parameter("keep_alive", "0").value)
        self.num_ctx = int(self.declare_parameter("num_ctx", 4096).value)
        self.open_vocab = bool(self.declare_parameter("open_vocab", True).value)
        self.br = CvBridge()
        self.rgb = self.info = None
        self.create_subscription(CameraInfo, "/overhead/camera_info", lambda m: setattr(self, "info", m), 1,
                                 callback_group=self.cb)
        self.create_subscription(Image, "/overhead/image", lambda m: setattr(self, "rgb", m), 1,
                                 callback_group=self.cb)
        self.det = self.create_client(DetectObjects, "/detect_objects", callback_group=self.cb)
        self.opn = self.create_client(DetectOpen, "/detect_open", callback_group=self.cb)
        self.create_service(QueryScene, "/query_scene", self.on_query, callback_group=self.cb)
        self.get_logger().info(f"/query_scene ready (model {self.model}, keep_alive {self.keep_alive})")

    def _wait(self, fut, timeout):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def verified_objects(self):
        """{name: (x, y, top)} from the detectors (the only source of truth for what is on the table)."""
        out = {}
        if self.det.wait_for_service(timeout_sec=2.0):
            req = DetectObjects.Request(); req.color = "all"
            r = self._wait(self.det.call_async(req), 10.0)
            if r is not None and r.success:
                for i, p in zip(r.ids, r.poses):
                    out.setdefault(i.split("_")[0], (p.pose.position.x, p.pose.position.y, p.pose.position.z))
        if self.open_vocab and self.opn.wait_for_service(timeout_sec=0.5):
            q = DetectOpen.Request(); q.labels = list(EXTRA_LABELS.values())
            r = self._wait(self.opn.call_async(q), 120.0)
            by_label = {v: k for k, v in EXTRA_LABELS.items()}
            if r is not None:
                for lab, p in zip(r.found_labels, r.poses):
                    key = by_label.get(lab)
                    x, y, top = p.pose.position.x, p.pose.position.y, p.pose.position.z
                    if key is None or key in out:
                        continue
                    if any(math.hypot(x - cx, y - cy) < DUP_RADIUS for n, (cx, cy, _) in out.items() if n in COLORS):
                        continue
                    if not (TABLE_TOP + 0.02 < top < TABLE_TOP + 3 * CUBE + 0.03):
                        continue            # probably the arm
                    out[key] = (x, y, top)
        return out

    def on_query(self, req, res):
        q = req.question.strip()
        if not q:
            res.success, res.message = False, "empty question"
            return res
        if self.rgb is None or self.info is None:
            res.success, res.message = False, "no camera data yet"
            return res
        objs = self.verified_objects()
        if not objs:
            res.success, res.message = False, "no objects detected (is the arm at home? are the detectors running?)"
            return res
        from PIL import Image as PILImage
        pil = PILImage.fromarray(self.br.imgmsg_to_cv2(self.rgb, "rgb8"))
        marks = {n: V.project(x, y, top, self.info.k) for n, (x, y, top) in objs.items()}
        img = V.annotate(pil, marks)
        try:
            img.save(DEBUG_IMAGE)
            res.image_path = DEBUG_IMAGE
        except Exception:
            pass
        names = [n for n in V.NAMES if n in objs]
        t0 = time.time()
        try:
            a = V.ask(self.host, self.model, q, img, names, self.keep_alive, self.num_ctx)
        except Exception as e:
            res.success, res.message = False, f"VLM call failed (ollama running? `ollama pull {self.model}`): {e}"
            return res
        self.get_logger().info(f"Q: {q} -> {a['objects']} amb={a['ambiguous']} '{a['answer']}' "
                               f"({time.time() - t0:.1f}s, dropped {a['dropped']})")
        if not a["ok"]:
            res.success, res.message = False, f"VLM reply was not valid JSON: {a['answer']}"
            return res
        res.success, res.message = True, "ok" + (f" (ignored non-detected: {a['dropped']})" if a["dropped"] else "")
        res.answer, res.objects = a["answer"], a["objects"]
        res.ambiguous, res.clarification = a["ambiguous"], a["clarification"]
        return res


def main():
    rclpy.init()
    node = VlmNode()
    ex = MultiThreadedExecutor(6)
    ex.add_node(node)
    ex.spin()
