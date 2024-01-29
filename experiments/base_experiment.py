import numpy as np
import os
from pathlib import Path
import sys
import torch


def setup_experiment(seed: int, data_path: str, log_dir: str, device: int = 0):
    experiment_dir = Path(__file__).parent.resolve()
    root_dir = Path(os.path.abspath(f"{str(experiment_dir)}/.."))
    sys.path.append(str((root_dir / "src").resolve()))

    data_path = root_dir / data_path
    log_dir = root_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Using directories:")
    print("root_dir:", root_dir)
    print("data_dir:", data_path)
    print("log_dir:", log_dir)
    print("========================================")
    print(
        "device:",
        device,
    )

    return root_dir, data_path, log_dir, device
