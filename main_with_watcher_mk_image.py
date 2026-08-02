import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import get_config
from watchFolder_Thread import start_watching


APP_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = APP_DIR / "gk2a_image_worker.py"

config = get_config()
MAX_WORKERS = max(1, int(os.getenv("GK2A_IMAGE_MAX_WORKERS", "2")))
RETRY_COUNT = int(os.getenv("GK2A_IMAGE_RETRY_COUNT", "1"))
RETRY_DELAY = float(os.getenv("GK2A_IMAGE_RETRY_DELAY", "30"))
MONO_ALPHA_MODE = os.getenv("GK2A_MONO_ALPHA_MODE", "A").upper()

if MONO_ALPHA_MODE not in ("A", "B", "C"):
    raise ValueError(f"Unsupported GK2A_MONO_ALPHA_MODE: {MONO_ALPHA_MODE}")

print(f"Running in {config.ENV} mode")
print(f"OUT_PATH = {config.OUT_PATH}")
print(f"WATCH_PATH = {config.WATCH_PATH}")
print(f"GK2A_IMAGE_MAX_WORKERS = {MAX_WORKERS}")
print(f"GK2A_MONO_ALPHA_MODE = {MONO_ALPHA_MODE}")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def run_worker_once(nc_file):
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--file",
        str(nc_file),
        "--mono-alpha-mode",
        MONO_ALPHA_MODE,
    ]
    return subprocess.run(cmd, cwd=str(APP_DIR), env=os.environ.copy()).returncode


def run_worker_with_retries(nc_file):
    for attempt in range(1, RETRY_COUNT + 2):
        print(f"start image worker attempt {attempt}/{RETRY_COUNT + 1}: {nc_file}", flush=True)
        returncode = run_worker_once(nc_file)
        if returncode == 0:
            print(f"completed image worker: {nc_file}", flush=True)
            return

        print(f"image worker failed: {nc_file} (returncode={returncode})", flush=True)
        if attempt <= RETRY_COUNT:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"image worker failed after retries: {nc_file}")


def on_worker_done(future):
    try:
        future.result()
    except Exception as exc:
        print(f"image worker job failed: {exc}", flush=True)


def callback(nc_file):
    print(f"queue image worker: {nc_file}", flush=True)
    future = executor.submit(run_worker_with_retries, nc_file)
    future.add_done_callback(on_worker_done)


if __name__ == "__main__":
    try:
        start_watching(config.WATCH_PATH, None, callback)
    finally:
        print("waiting for queued image workers to finish...", flush=True)
        executor.shutdown(wait=True)
