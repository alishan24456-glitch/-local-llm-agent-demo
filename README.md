# Local LLM Agent Demo

**CPU-only. This demo runs a small language model 100% locally on a normal Windows PC — no cloud, no GPU, no API key.**

Built by **Abdul Hanan** (Computer Vision developer, Python / OpenCV / YOLOv8) as a hands-on demo for a Local LLM / AI Agent Engineer application. It is my first local-LLM build: a tiny agent that takes a plain-English request, picks one of two tools, runs it, and lets the local model write the final answer.

- **Runtime:** Ollama (CPU)
- **Model:** Qwen2.5-1.5B-Instruct, quantised Q4_K_M (`qwen2.5:1.5b`, ~1 GB)
- **Agent:** one Python file, standard library only for the LLM part (`urllib` → `http://localhost:11434`)

## The two tools

| Tool | What it does |
|---|---|
| `summarise_log` | Summarises a factory inspection log — a built-in sample log, or your own `.txt` file |
| `detect_objects` | Runs YOLOv8 on one image (chosen with a file picker) and returns object counts, then the local model describes the result in one sentence |

Routing is deliberately simple and honest: keywords about logs go to Tool 1, keywords about images go to Tool 2, and anything else is a normal chat question to the local model.

## Requirements

- Windows PC
- Python 3.12
- Ollama installed — https://ollama.com/download
- The model downloaded once:

```
ollama pull qwen2.5:1.5b
```

Optional, only for Tool 2 (object detection):

```
pip install ultralytics
```

If `ultralytics` is not installed, Tool 2 prints this instruction and the rest of the demo still works.

## How to run (Windows CMD)

1. Make sure Ollama is running (after install it normally runs by itself).
2. Open CMD in the folder where `local_llm_agent_demo.py` is saved.
3. Run the demo:

```
python local_llm_agent_demo.py
```

4. Type a request in plain English, for example:
   - `summarise the inspection log`
   - `detect objects in an image`
   - any normal question, e.g. `what is a conveyor belt?`
5. Type `/bye` to quit.

### Speed test

```
python local_llm_agent_demo.py --speed-test
```

(or type `/speed` inside the chat). This runs one fixed prompt 3 times and prints the measured time and tokens/second from that actual run. Nothing is pre-filled or estimated by hand — the numbers come from Ollama's own response data.

## My results

Measured on my own PC with the speed test above (run Wed 18 Nov 2026 ~21:00 PKT, Qwen2.5-1.5B on CPU via Ollama).

| Run | Time (s) | Tokens/sec |
|---|---|---|
| 1 | 4.89 | 13.2 |
| 2 | 4.96 | 11.6 |
| 3 | 5.90 | 9.2 |

My PC: *(CPU / RAM — fill in your own specs)*

## One failure I hit, and the fix

A long log (or a long, unclear request) can go past the small model's context window, and the answer then gets cut off or misses the point. The fix used in this demo: logs are trimmed to the first 4,000 characters with a clear on-screen note, and requests work best as one short, clear English sentence.

## Limitations (honest notes)

- This is a **small 1.5B model running on CPU only** — it is slow compared to a GPU server, and its answers can be simple or sometimes wrong.
- Example from my own testing: it misunderstood a request written in Roman Urdu and answered something generic instead — it works much better with short, clear English.
- It does not "remember" anything between separate runs, and it is not fine-tuned or trained by me — it is the stock quantised model, run locally.
- GPU inference (e.g. vLLM on a GPU machine) is a **next step I want to learn**, not experience I am claiming here.

## Architecture idea (where this could go)

```
Slack message
    -> Router agent (local LLM: what is being asked?)
        -> Log agent      (file tool: read + summarise inspection logs)
        -> Inspection agent (YOLO tool: check an image, return counts)
    -> Reviewer agent (checks the draft answer against the tool output)
    -> Reply posted back to Slack
    -> Selective long-term memory: only verified facts are saved,
       each with its source and date
```

## Files

- `local_llm_agent_demo.py` — the whole demo, one file
- `README.md` — this file
