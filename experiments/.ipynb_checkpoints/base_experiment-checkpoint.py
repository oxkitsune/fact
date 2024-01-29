import os
from pathlib import Path
import sys

def setup_experiment(device_index: int = 0, seed: int, data_path: str, log_dir: str):
    experiment_dir = Path(__file__).parent.resolve()
    root_dir = Path(os.path.abspath(f"{str(experiment_dir)}/.."))
    sys.path.append(str((root_dir / "src").resolve()))

    print("root_dir:", root_dir)
    

    