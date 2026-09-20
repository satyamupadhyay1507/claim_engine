import runpy
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Run the Streamlit app in src/ui/app.py
target_app = root_dir / "src" / "ui" / "app.py"
runpy.run_path(str(target_app), run_name="__main__")
