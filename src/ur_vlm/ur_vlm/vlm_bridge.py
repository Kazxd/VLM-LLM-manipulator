"""Chat with the robot, VLM-assisted: the LLM plans, the VLM resolves visual references and ambiguity.

  ros2 run ur_vlm vlm_bridge --ros-args -p model:=qwen3:4b-instruct
Needs: sim, detect_objects, skill_server, vlm_node (+ detect_open for can/ball/block), `ollama serve`
with both the LLM and qwen3-vl:4b pulled. Run INSTEAD of llm_bridge.

The LLM gets one extra tool, ask_scene(question). The reply (object names from verified detections, or a
clarification question) comes back as the tool result; every pick/place still goes through the existing validators.
"""
import threading, rclpy
from rclpy.executors import MultiThreadedExecutor
from ur_interfaces.srv import QueryScene
from ur_interfaces.action import MoveHome
from ur_llm_bridge import agent
from ur_llm_bridge.llm_bridge import LLMBridge

ASK_TOOL = agent._fn(
    "ask_scene",
    "Ask the vision model about the table when the request describes an object by position or relation instead of "
    "name, e.g. 'which object is closest to the can?', 'the leftmost object'. Returns object names.",
    {"question": {"type": "string"}}, ["question"])
TOOLS = agent.TOOLS + [ASK_TOOL]

VLM_RULES = """
Vision: if the request names an object by position or relation ("the one next to the can", "the leftmost object", \
"the thing closest to the robot"), call ask_scene(question) FIRST, then use the returned object names with pick_object / \
place_object. If ask_scene returns ambiguous=true, ask the user its clarification and call no other tool. \
Never guess a name. Trust the returned names, not any coordinates."""

_orig_validate = agent.validate


def validate(name, args):
    if name == "ask_scene":
        if not isinstance(args, dict):
            return None, "arguments must be a JSON object"
        q = str(args.get("question", "")).strip()
        if not 3 <= len(q) <= 300:
            return None, "question must be 3-300 characters"
        return {"question": q}, None
    return _orig_validate(name, args)


agent.validate = validate        # run_agent looks validate() up in the agent module


def new_history():
    return [{"role": "system", "content": agent.SYSTEM + agent.EXAMPLES + VLM_RULES}]


class VlmBridge(LLMBridge):
    def __init__(self, name="vlm_bridge"):
        super().__init__(name)
        self.vlm_cli = self.create_client(QueryScene, "/query_scene")

    def ask_scene(self, a):
        ok, msg = self.call_action(self.home_cli, MoveHome.Goal())     # arm out of the camera's view
        if not ok:
            return {"success": False, "message": "could not clear the camera view: " + msg}
        if not self.vlm_cli.wait_for_service(timeout_sec=3.0):
            return {"success": False, "message": "/query_scene not available (ros2 run ur_vlm vlm_node)"}
        req = QueryScene.Request(); req.question = a["question"]
        fut = self.vlm_cli.call_async(req)
        if not self._wait(fut, 240.0):          # the first call loads the VLM
            return {"success": False, "message": "VLM timed out"}
        r = fut.result()
        if not r.success:
            return {"success": False, "message": r.message}
        out = {"success": True, "answer": r.answer, "objects": list(r.objects), "ambiguous": r.ambiguous}
        if r.ambiguous:
            out["clarification"] = r.clarification
        return out

    def execute(self, name, a):
        if name == "ask_scene":
            return self.ask_scene(a)
        return super().execute(name, a)


def main():
    rclpy.init()
    node = VlmBridge()
    ex = MultiThreadedExecutor(4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    history = new_history()
    chat = lambda msgs: agent.ollama_chat(node.host, node.model, msgs, TOOLS, no_think=not node.think)
    print(f"VLM-assisted bridge ready (LLM {node.model}). Type a command, or 'quit'.")
    while rclpy.ok():
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if text:
            print("robot> " + agent.run_agent(history, text, node.execute, chat, node.scene_summary))
    rclpy.shutdown()
