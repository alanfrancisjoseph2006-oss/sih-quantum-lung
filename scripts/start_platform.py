"""
Start the full QuantumLung platform: FastAPI backend + Next.js frontend.

  python scripts/start_platform.py            # dev (hot reload)
  python scripts/start_platform.py --prod     # build + serve the production bundle

Backend  -> http://127.0.0.1:8000   (API + legacy static frontend)
Frontend -> http://localhost:3000   (clinical platform)
"""
from __future__ import annotations

import argparse
import atexit
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT))

PROCS: list[subprocess.Popen] = []


def shutdown() -> None:
    for p in PROCS:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass


atexit.register(shutdown)


def wait_for(url: str, timeout: float = 180.0) -> bool:
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True          # responding, even if not 200
        except Exception:
            time.sleep(1.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true", help="build and serve the production bundle")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        print("npm not found on PATH - install Node.js 20+ to run the frontend.")
        return 1
    if not (WEB / "node_modules").exists():
        print("Installing frontend dependencies (first run only)...")
        subprocess.run([npm, "install"], cwd=WEB, check=True)

    from src.config import hardware_summary
    from src.inference.engine import engine_status

    hw = hardware_summary()
    st = engine_status()
    print("=" * 70)
    print(" QuantumLung - AI-Assisted Lung Screening & Radiology Decision Support")
    print(" SIH 2026 | PS 26139 | Team CodeNova")
    print("=" * 70)
    print(f" device   : {hw['device']}" +
          (f" ({hw['gpu_name']}, {hw['vram_gb']} GB)" if hw["cuda"] else " (CPU mode)"))
    if st.get("ready"):
        print(f" models   : {st['backbone']} -> {st['primary_model']} "
              f"({st['n_qubits']} qubits)")
    else:
        print(f" models   : NOT READY - {st.get('error')}")
        print("            run: python scripts/run_all.py")
    print("=" * 70)

    # ---- backend ----
    print("\nStarting API on http://127.0.0.1:8000 ...")
    PROCS.append(subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT))
    if not wait_for("http://127.0.0.1:8000/api/health"):
        print("Backend failed to start."); return 1
    print("  API ready.")

    # ---- frontend ----
    if args.prod:
        print("\nBuilding the production frontend (this takes a minute) ...")
        r = subprocess.run([npm, "run", "build"], cwd=WEB)
        if r.returncode != 0:
            print("Frontend build failed."); return 1
        cmd = [npm, "run", "start"]
    else:
        cmd = [npm, "run", "dev"]

    print("\nStarting the clinical platform on http://localhost:3000 ...")
    PROCS.append(subprocess.Popen(cmd, cwd=WEB))
    if not wait_for("http://localhost:3000"):
        print("Frontend failed to start."); return 1

    print("\n" + "=" * 70)
    print("  Clinical platform : http://localhost:3000")
    print("  API               : http://127.0.0.1:8000")
    print("  API docs          : http://127.0.0.1:8000/docs")
    print("=" * 70)
    print("\n  AI-assisted research and clinical decision-support tool.")
    print("  Not a replacement for professional medical diagnosis.\n")
    print("Press Ctrl+C to stop both services.")

    if not args.no_browser:
        webbrowser.open("http://localhost:3000")

    try:
        while all(p.poll() is None for p in PROCS):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
