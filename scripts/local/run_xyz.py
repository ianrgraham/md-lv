import glob
import subprocess
import numpy as np

vs = np.linspace(1, 11, 11)

files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/xyz/equil_*")

for i, f in enumerate(files):
    print(i)
    for v in vs:
        command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --potential hertz --time 10.0 \
            --init-config {f} --temp 0.01 --vscale {v} --seed {200000+i} --dt 1e-3 --out-time 1e-1 \
            --dir /home/ian/Documents/Data/MD_LV_paper_data/xyz'

        input = command.split()
        subprocess.run(input)
    break