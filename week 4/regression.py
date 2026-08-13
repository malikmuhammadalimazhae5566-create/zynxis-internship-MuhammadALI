from pathlib import Path
import runpy

project_script = Path(__file__).resolve().parent / "Regression_Project" / "regression.py"
runpy.run_path(str(project_script), run_name="__main__")
