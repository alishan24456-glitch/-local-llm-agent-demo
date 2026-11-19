"""
local_llm_agent_demo.py

A small LOCAL LLM agent demo, built by Abdul Hanan for a job application.

What it does:
  - Talks to a local model running in Ollama (no internet, no cloud, CPU-only).
  - Model: qwen2.5:1.5b  (Qwen2.5-1.5B-Instruct, small and fast on a normal PC)
  - The agent has TWO tools:
      Tool 1 "summarise_log"  : summarises a factory inspection log (.txt)
      Tool 2 "detect_objects" : runs YOLOv8 object detection on ONE image
  - You type a request in plain English, the script picks the right tool,
    runs it, and then asks the local model to write the final answer.

How to run (Windows CMD):
  1. Start Ollama (it normally runs by itself after install).
  2. ollama pull qwen2.5:1.5b        (only needed the first time)
  3. python local_llm_agent_demo.py
  Extra:  python local_llm_agent_demo.py --speed-test

Type /bye to quit. Type /speed inside the chat to run the speed test.
Only the Python standard library is used for the LLM part (urllib).
Tool 2 needs the optional "ultralytics" package - see the message it
prints if that package is not installed.
"""

import json
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------- settings

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:1.5b"

# A small built-in factory inspection log, used when you do not have
# your own .txt log file handy.
SAMPLE_LOG = """Factory Inspection Log - Line 2 - 18 Nov 2026
08:00  Shift start. Conveyor belt running normally.
08:15  Bottle cap check: 3 bottles with loose caps removed.
08:40  Temperature of oven 2 slightly high (212 C, limit 210 C). Adjusted.
09:05  Worker reported a jam at the labelling machine. Cleared in 6 minutes.
09:30  Quality check: 2 cracked bottles found and rejected.
10:00  Break. Line stopped for 10 minutes as planned.
10:20  Camera 3 image blurry. Lens cleaned, image clear again.
10:45  Box count mismatch: 48 boxes counted, 50 expected. Recount started.
11:10  Recount finished: 50 boxes confirmed, counting error only.
11:30  End of morning inspection. No safety incidents.
"""

# ---------------------------------------------------------------- ollama


def check_ollama():
    """Return True if the Ollama server answers, False if it does not."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def ask_model(prompt):
    """
    Send one prompt to the local model and return (answer_text, info_dict).
    info_dict may contain 'eval_count' and 'eval_duration' from Ollama,
    which the speed test uses to calculate tokens per second.
    Raises RuntimeError with a friendly message if something goes wrong.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError:
        raise RuntimeError(
            "Could not reach Ollama at " + OLLAMA_URL + ".\n"
            "Please make sure Ollama is running, then try again.\n"
            "Tip: open a new CMD window and type:  ollama run qwen2.5:1.5b"
        )
    except Exception as e:
        raise RuntimeError("Problem talking to the local model: " + str(e))

    return result.get("response", "").strip(), result


# ---------------------------------------------------------------- tool 1


def tool_summarise_log():
    """Tool 1: summarise a factory inspection log with the local model."""
    print()
    print("TOOL 1 - Summarise a factory inspection log")
    print("Press ENTER to use the built-in sample log,")
    path = input("or type the full path of your own .txt log file: ").strip()

    if path == "":
        log_text = SAMPLE_LOG
        print("(Using the built-in sample log.)")
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                log_text = f.read()
        except OSError:
            print("Sorry, I could not open that file. Using the sample log.")
            log_text = SAMPLE_LOG

    # Small models have a small context window. If the log is very long,
    # we keep only the first part and say so honestly instead of failing.
    MAX_CHARS = 4000
    if len(log_text) > MAX_CHARS:
        print("(Log is long, so only the first part was sent to the model.)")
        log_text = log_text[:MAX_CHARS]

    prompt = (
        "Here is a factory inspection log.\n"
        "Summarise it in exactly 3 short bullet points. "
        "Mention any problems or warnings you can see.\n\n"
        "LOG:\n" + log_text + "\n\nSummary:"
    )
    print("Asking the local model, please wait...")
    answer, _info = ask_model(prompt)
    print()
    print("--- Log summary ---")
    print(answer)
    print("-------------------")


# ---------------------------------------------------------------- tool 2


def pick_image_file():
    """
    Open a tkinter file-chooser for an image.
    If the chooser is not available, fall back to typing the path.
    Returns the chosen path, or "" if nothing was chosen.
    """
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose one image for object detection",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if path:
            return path
        print("No file chosen in the window.")
    except Exception:
        print("(File chooser window not available.)")

    return input("Type the full path of the image file: ").strip()


def tool_detect_objects():
    """Tool 2: run YOLOv8 on one image, if ultralytics is installed."""
    print()
    print("TOOL 2 - Detect objects in one image (YOLOv8)")

    # Import here, so the rest of the demo still works without it.
    try:
        from ultralytics import YOLO
    except ImportError:
        print()
        print("The 'ultralytics' package is not installed, so Tool 2")
        print("cannot run yet. The rest of the demo still works.")
        print("To enable Tool 2, close this program and run this command:")
        print()
        print("    pip install ultralytics")
        print()
        print("Then start this program again.")
        return

    image_path = pick_image_file()
    if image_path == "":
        print("No image given, going back to the menu.")
        return

    print("Running YOLOv8 on the image, please wait...")
    try:
        model = YOLO("yolov8n.pt")  # downloads the small model once
        results = model(image_path, verbose=False)
    except Exception as e:
        print("Detection failed: " + str(e))
        return

    # Count how many of each object class was found.
    counts = {}
    names = results[0].names
    for box in results[0].boxes:
        label = names[int(box.cls[0])]
        counts[label] = counts.get(label, 0) + 1

    print()
    print("--- Detection result ---")
    if not counts:
        print("No objects were detected in this image.")
        return
    for label in sorted(counts):
        print("  " + label + ": " + str(counts[label]))
    print("------------------------")

    # Let the local model turn the raw counts into one friendly sentence.
    counts_text = ", ".join(
        str(counts[k]) + " x " + k for k in sorted(counts)
    )
    prompt = (
        "An object detector found these objects in one image: "
        + counts_text
        + ". Write ONE short friendly sentence describing this result."
    )
    try:
        answer, _info = ask_model(prompt)
        print("Model says: " + answer)
    except RuntimeError as e:
        print(str(e))


# ------------------------------------------------------------ speed test


def run_speed_test():
    """
    Run one fixed prompt 3 times and print a small results table.
    All numbers are measured from THIS run - nothing is hard-coded.
    """
    prompt = "Explain in two sentences what a factory inspection log is."
    print()
    print("SPEED TEST - same prompt, 3 runs, model: " + MODEL_NAME)
    print("Prompt: " + prompt)
    print()
    print("Run | Time (s) | Tokens/sec")
    print("----+----------+-----------")

    for run_number in (1, 2, 3):
        start = time.time()
        try:
            _answer, info = ask_model(prompt)
        except RuntimeError as e:
            print(str(e))
            return
        elapsed = time.time() - start

        # Ollama reports eval_count (tokens produced) and eval_duration
        # (time spent producing them, in nanoseconds). Use them if present.
        eval_count = info.get("eval_count")
        eval_duration = info.get("eval_duration")
        if eval_count and eval_duration:
            tokens_per_sec = eval_count / (eval_duration / 1_000_000_000)
        elif eval_count and elapsed > 0:
            tokens_per_sec = eval_count / elapsed
        else:
            tokens_per_sec = 0.0

        print(
            " "
            + str(run_number)
            + "  |   "
            + format(elapsed, ".2f")
            + "   |   "
            + format(tokens_per_sec, ".1f")
        )

    print()
    print("Tip: copy your 3 rows into the 'My results' table in README.md.")


# -------------------------------------------------------------- routing


def route_request(request):
    """
    Very simple, honest routing:
    words about logs -> Tool 1, words about images -> Tool 2,
    anything else -> a normal chat answer from the local model.
    """
    text = request.lower()

    log_words = ("log", "summarise", "summarize", "summary", "inspection")
    image_words = ("detect", "image", "photo", "picture", "object", "yolo")

    if any(word in text for word in log_words):
        tool_summarise_log()
    elif any(word in text for word in image_words):
        tool_detect_objects()
    else:
        print("Asking the local model, please wait...")
        try:
            answer, _info = ask_model(request)
            print()
            print("Model: " + answer)
        except RuntimeError as e:
            print(str(e))


# ----------------------------------------------------------------- main


def main():
    # --speed-test flag: run the test once and finish.
    if "--speed-test" in sys.argv:
        if not check_ollama():
            print("ERROR: Ollama is not running at " + OLLAMA_URL)
            print("Start Ollama first, then run this file again.")
            return
        run_speed_test()
        return

    print("=" * 60)
    print(" Local LLM Agent Demo - Abdul Hanan")
    print(" Model: " + MODEL_NAME + " (local, CPU-only, via Ollama)")
    print("=" * 60)

    if not check_ollama():
        print()
        print("ERROR: Ollama is not running at " + OLLAMA_URL)
        print("Please start Ollama, then run this file again.")
        print("Tip: open a new CMD window and type:  ollama run qwen2.5:1.5b")
        return

    print()
    print("The local model is ready. Things you can type:")
    print('  - "summarise the inspection log"   -> Tool 1')
    print('  - "detect objects in an image"     -> Tool 2')
    print("  - any normal question              -> the model answers")
    print("  - /speed   run the speed test")
    print("  - /bye     quit")
    print()

    while True:
        try:
            request = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if request == "":
            continue
        if request.lower() == "/bye":
            print("Bye!")
            break
        if request.lower() == "/speed":
            run_speed_test()
            continue

        route_request(request)
        print()


if __name__ == "__main__":
    main()
