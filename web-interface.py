#!/usr/bin/env python3

import os
import signal
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))

backend_cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "backend.api.app:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]

frontend_cmd = ["npm", "run", "dev"]


def main():
    processes = []

    try:
        print("🚀 Starting backend...")
        backend = subprocess.Popen(
            backend_cmd,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": ROOT},
        )
        processes.append(backend)

        time.sleep(2)

        print("🚀 Starting frontend...")
        frontend = subprocess.Popen(
            frontend_cmd,
            cwd=os.path.join(ROOT, "frontend"),
        )
        processes.append(frontend)

        print("\n===================================")
        print("Turbo-LLM is running!")
        print("Frontend : http://localhost:3000")
        print("Backend  : http://localhost:8000")
        print("Press Ctrl+C to stop everything.")
        print("===================================\n")

        while True:
            time.sleep(1)

            for p in processes:
                if p.poll() is not None:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nStopping...")

        for p in processes:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)

        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

        print("Done.")


if __name__ == "__main__":
    main()