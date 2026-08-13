from pathlib import Path
import os
import runpy
import sys


def find_project_dir(start: Path) -> Path:
    candidates = [
        start / "NLP_Pipeline_Zynxis",
        start,
        start.parent / "NLP_Pipeline_Zynxis",
    ]

    for candidate in candidates:
        if (candidate / "main.py").exists() and (candidate / "src").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError("Could not find the NLP_Pipeline_Zynxis project directory.")


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = find_project_dir(ROOT)
MAIN_FILE = PROJECT_DIR / "main.py"

if not MAIN_FILE.exists():
    raise FileNotFoundError(f"Could not find the project entry point at {MAIN_FILE}")

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(ROOT))
os.chdir(PROJECT_DIR)
print(f"Launching NLP pipeline from: {PROJECT_DIR}")
runpy.run_path(str(MAIN_FILE), run_name="__main__")
