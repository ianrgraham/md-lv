import glob
import subprocess
import numpy as np
import sys


As = np.linspace(0.1, 0.3, 15)
as_str = ",".join([str(a) for a in As])

vs = np.logspace(0, 1, 11)

idx = int(sys.argv[1]) + 22999

pot = "lj"

files = glob.glob(f"/home/igraham/data/MD_LV_paper_data/equil_n25/equil_*{pot}*")

assert(len(files) == 1)
f = files[0]

for vscale in vs:

    command = f'/home/igraham/Documents/md-lv/target/release/md-lv --unwrap --potential {pot} \
        --init-config {f} --temp 0.1 --vscale {vscale} --seed {idx} --dt 1e-3 --out-time 1e-2 \
        --dir /home/igraham/data/MD_LV_paper_data/{pot}_val_n25 --time 10 \
        gen-variant --realizations 10000 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

    input = command.split()
    subprocess.run(input)