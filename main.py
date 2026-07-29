from pathlib import Path
import os
import sys


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "qgan" / "src"))
os.chdir(REPO_ROOT)

from qgan_v2.main import main


if __name__ == "__main__":
    main()
