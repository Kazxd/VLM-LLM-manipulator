"""Ask the VLM one question and print the result.

  ros2 run ur_vlm vlm_check "which object is closest to the can?"
Needs vlm_node running and the arm at home. The annotated image is saved to /tmp/vlm_last.png.
"""
import sys, rclpy
from rclpy.node import Node
from ur_interfaces.srv import QueryScene


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("__")]
    q = " ".join(args) or "which object is the leftmost?"
    rclpy.init()
    n = Node("vlm_check")
    cli = n.create_client(QueryScene, "/query_scene")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("/query_scene not available (ros2 run ur_vlm vlm_node)")
        rclpy.shutdown()
        return
    req = QueryScene.Request(); req.question = q
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=240.0)
    r = fut.result()
    if r is None:
        print("timed out")
    else:
        print(f"success={r.success} ({r.message})\nanswer: {r.answer}\nobjects: {list(r.objects)}\n"
              f"ambiguous: {r.ambiguous} {r.clarification}\nimage: {r.image_path}")
    rclpy.shutdown()
