import numpy as np
import os
from pathlib import Path
import sys
import torch

from typing import List, Dict


class ExperimentRunner:
    """This class is used to run experiments with different seeds and parameters.
    It provides a method `runs` which can be used to iterate over all the permutations of seeds and parameters, for a given experiment.
    """

    def __init__(
        self,
        experiment_name: str,
        seeds: List[int],
        data_path: str,
        log_dir: str,
        device: torch.device,
        params: List[Dict],
    ):
        """Initialize a new ExperimentRunner.

        Args:
            experiment_name (str): The name of the experiment, used for creating the log directory.
            seeds (List[int]): A list of seeds to run the experiment with.
            data_path (str): The path to the data directory.
            log_dir (str): The path to the root of the log directory, the experiment log will be created in a subdirectory of this.
            device (torch.device): The device to run the experiment on.
            params (List[Dict]): A list of dictionaries, each dictionary contains the parameters for a single run of the experiment.
        """
        assert len(seeds) > 0, "Atleast a single seed has to be specified!"
        assert len(params) > 0, "Atleast a single param has to be specified!"

        known_params = set()
        for key in params[0].keys():
            known_params.add(key)

        for param in params:
            for key in param.keys():
                assert (
                    key in known_params
                ), f"Unknown param {key}! Known params are {known_params}"

        self.experiment_name = experiment_name
        self.seeds = seeds
        self.params = params
        self.device = device

        self.experiment_dir = Path(__file__).parent.resolve()
        self.root_dir = Path(os.path.abspath(f"{str(self.experiment_dir)}/.."))

        # add the source dir to the system path
        sys.path.append(str((self.root_dir / "src").resolve()))

        # set up the base directories
        self.data_path = self.root_dir / data_path
        self.log_dir = self.root_dir / log_dir

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def runs(self):
        """Generate the runs for the experiment, with all the permutations of seeds and parameters.
        It will iterate over all parameters for each seed, running all experiments for a single seed before moving to the next seed.

        Yields:
            Tuple[int, Path, torch.device, Dict]: A tuple containing the seed, log directory, device and the parameters for the run.
        """
        for seed in self.seeds:
            for params in self.params:
                # set the random seed
                self._set_seed(seed)

                # create the param string
                param_str = "_".join(
                    [f"{key}_{value}" for key, value in params.items()]
                )

                # create the log directory
                log_dir = self.log_dir / f"{self.experiment_name}_{seed}_{param_str}"
                log_dir.mkdir(parents=True, exist_ok=True)

                yield seed, log_dir, self.device, params


def setup_experiment(seed: int, data_path: str, log_dir: str, device: int = 0):
    """Set up the environment for running an experiment.

    Args:
        seed (int): The random seed to use for the experiment.
        data_path (str): The path to the data directory.
        log_dir (str): The path to the root of the log directory, the experiment log will be created in a subdirectory of this.
        device (int, optional): The device to run the experiment on. Defaults to 0.

    Returns:
        Tuple[Path, Path, Path, torch.device]: A tuple containing the root directory, data directory, log directory and the device to run the experiment on.
    """
    experiment_dir = Path(__file__).parent.resolve()
    root_dir = Path(os.path.abspath(f"{str(experiment_dir)}/.."))
    sys.path.append(str((root_dir / "src").resolve()))

    data_path = root_dir / data_path
    log_dir = root_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

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


def setup_evaluation(seed: int, data_path: str, model_dir: str, device: int = 0):
    """Set up the environment for running an evaluation.

    Args:
        seed (int): The random seed to use for the evaluation.
        data_path (str): The path to the data directory.
        model_dir (str): The path to the model directory.
        device (int, optional): The device to run the evaluation on. Defaults to 0.

    Returns:
        Tuple[Path, Path, Path, torch.device]: A tuple containing the root directory, data directory, model directory and the device to run the evaluation on.
    """
    experiment_dir = Path(__file__).parent.resolve()
    root_dir = Path(os.path.abspath(f"{str(experiment_dir)}/.."))
    sys.path.append(str((root_dir / "src").resolve()))

    data_path = root_dir / data_path
    model_dir = root_dir / model_dir
    assert model_dir.exists(), "Model file couldn't be resolved! Ensure it exists!"

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("Using directories:")
    print("root_dir:", root_dir)
    print("data_dir:", data_path)
    print("model_dir:", model_dir)
    print("========================================")
    print(
        "device:",
        device,
    )

    return root_dir, data_path, model_dir, device
