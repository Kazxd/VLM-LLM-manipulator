# Handoff: VLM-assisted architecture (new project, forked from ur_llm_ws / HANDOFF6)

Everything in HANDOFF6.md still applies (sim, skills, detectors, benchmark, troubleshooting). Added:

## New / changed files
- src/ur_interfaces/srv/QueryScene.srv (NEW) and src/ur_interfaces/CMakeLists.txt (adds it)
- src/ur_vlm/ (NEW package): vlm_lib.py (ROS-free: projection, annotation, schema, parsing, Ollama call),
  vlm_node.py (/query_scene), vlm_bridge.py (LLM + ask_scene tool), vlm_check.py (CLI), test/test_vlm_lib.py (passes)
- No existing file in ur_llm_bridge / ur_skills / ur_perception was modified. vlm_bridge subclasses LLMBridge and
  patches agent.validate to accept ask_scene.

## Status
Offline test passes. NOT yet run live: vlm_node against Gazebo, vlm_bridge end to end.

## Known risks to check first
1. qwen3-vl:4b on a 960x720 annotated image: check /tmp/vlm_last.png is legible and answers are right.
2. Left/right/near/far convention in the prompt (image left = robot left). Verify with "which object is leftmost?".
3. VRAM: four models/processes on 6 GB. keep_alive 0 trades latency for memory.
4. ask_scene is not salvaged when the LLM writes it as text (TOOL_NAMES in agent.py is fixed at import).
5. Only the six known names exist: the VLM resolves references among them, it does not add new objects.

## Roadmap
1. Live check with vlm_check on 10 questions, record accuracy against ground truth (extend bench_lib with a `vlm` category).
2. Benchmark cases: "pick the object closest to the can", "put the leftmost object on the blue cube", ambiguous ones
   (two equally close objects must yield a clarification, nothing moved).
3. Post-action verification: after a place, ask the VLM "is red on top of blue?" and compare with detections.
4. Detector disagreement: if OWL-ViT and the VLM disagree on an object, report it instead of acting.
5. Cache the annotated image / VLM answer per scene state to cut latency.
