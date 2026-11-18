import os
import subprocess
for i in os.listdir("mfpa_fslr/configs"):
    if (i.find("train_wlasl") == -1): continue
    os.system(f"bash mfpa_fslr/scripts/run_conda.sh train --config configs/{i}")