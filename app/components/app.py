# Compatibility wrapper for SONOVA
# The primary entry point is app/app.py: streamlit run app/app.py

import sys
from pathlib import Path

# Add project root and app directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Forward execution to main app/app.py
main_app_path = PROJECT_ROOT / "app" / "app.py"
with open(main_app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, str(main_app_path), "exec"), globals(), locals())