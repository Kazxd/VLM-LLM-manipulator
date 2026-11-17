# UR5e + LLM + VLM (verified-detection architecture)

Fork of the UR5e local-LLM manipulator. Same simulator, skills, detectors and benchmark; one new layer: a
**vision-language model (qwen3-vl:4b via Ollama)** that interprets visual relations and ambiguity.

```
user ─▶ LLM planner (qwen3:4b) ──tool call──▶ validator ──▶ skill_server ──▶ MoveIt / Gazebo
              │  ▲                                                ▲
      ask_scene│  │names / clarification                           │
              ▼  │                                                 │
          vlm_node ◀── annotated overhead image ◀── verified detections (colour detector + OWL-ViT)
```

**Rules of the architecture**
1. Detectors are the only source of positions. The VLM never outputs coordinates.
2. The VLM sees the overhead image with each *detected* object's name drawn at its projected position (set-of-mark).
3. Its answer is schema-constrained to the detected names and validated again; unknown names are dropped.
4. The LLM plans with the existing validated tools plus `ask_scene(question)`.
5. Ambiguity (`ambiguous=true`) becomes a question to the user instead of a guess.

## Run
```bash
ollama pull qwen3-vl:4b
cd ~/CODE/ur_vlm_ws && colcon build --symlink-install
scripts/kill_sim.sh
ros2 launch ur_sim_bringup sim.launch.py
ros2 run ur_perception detect_objects
ros2 run ur_perception detect_open      # for can / ball / block
ros2 run ur_skills skill_server
ros2 run ur_vlm vlm_node
ros2 run ur_vlm vlm_bridge              # instead of llm_bridge
```
Try: "pick up the object closest to the can", "put the leftmost object on the blue cube".
Test the VLM alone: `ros2 run ur_vlm vlm_check "which object is closest to the can?"` (image: /tmp/vlm_last.png).
Offline test: `python3 src/ur_vlm/test/test_vlm_lib.py`.

## VRAM (6 GB)
Gazebo + OWL-ViT + qwen3:4b + qwen3-vl:4b do not all fit. `vlm_node` uses `keep_alive:=0` (unloads the VLM after each
query); expect a few seconds of reload per `ask_scene`. Use `-p keep_alive:=5m` if memory allows.

See HANDOFF_VLM.md for status and roadmap. License: MIT.
