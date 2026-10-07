from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bridge_tuning.cli import train_main
if __name__ == "__main__":
    train_main()
