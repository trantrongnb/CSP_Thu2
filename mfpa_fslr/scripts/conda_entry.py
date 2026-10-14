"""Entry point for the conda launcher; use its absolute interpreter path."""

import importlib
import os
import sys

action, args = sys.argv[1], sys.argv[2:]
if action == "check":
    print("Python:", sys.executable, flush=True)
    print("MPS pipe:", os.environ["CUDA_MPS_PIPE_DIRECTORY"], flush=True)
    import numpy
    import torch
    import yaml
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda, flush=True)
    print("NumPy:", numpy.__version__, "PyYAML:", yaml.__version__, flush=True)
    available = torch.cuda.is_available()
    print("CUDA available:", available, flush=True)
    if available:
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    sys.exit(0 if available else 1)

entries = {"train": ("mfpa.engine", "train_main"),
           "evaluate": ("mfpa.engine", "evaluate_main"),
           "episodes": ("mfpa.datasets.episode_sampler", "main")}
if action in {"train", "evaluate"} and not any(arg in {"-h", "--help"} for arg in args):
    if "--device" not in args and not any(arg.startswith("--device=") for arg in args):
        args.extend(["--device", "cuda"])
    device = next((arg.split("=", 1)[1] for arg in args if arg.startswith("--device=")), None)
    if "--device" in args and args.index("--device") + 1 < len(args):
        device = args[args.index("--device") + 1]
    if device and device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            sys.exit("CUDA is unavailable in this conda environment; run the 'check' command first.")
module, function = entries[action]
getattr(importlib.import_module(module), function)(args)
