"""Start the QuantumLung web application."""
import sys, webbrowser, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn
from src.config import hardware_summary
from src.inference.engine import engine_status

if __name__ == "__main__":
    print("=" * 66)
    print(" QuantumLung - Hybrid Quantum-Classical Lung Cancer Analysis")
    print(" SIH 2026 | PS 26139 | Team CodeNova")
    print("=" * 66)
    hw = hardware_summary()
    print(f" device   : {hw['device']}" + (f" ({hw['gpu_name']}, {hw['vram_gb']} GB)"
                                           if hw["cuda"] else " (no CUDA - CPU mode)"))
    st = engine_status()
    if st.get("ready"):
        print(f" models   : {st['backbone']} -> {st['primary_model']} ({st['n_qubits']} qubits)")
    else:
        print(f" models   : NOT READY - {st.get('error')}")
        print("            run scripts/run_all.py first")
    print(" open     : http://127.0.0.1:8000")
    print("=" * 66)
    threading.Timer(1.8, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, log_level="info")
