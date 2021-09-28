import glob
import subprocess
import numpy as np
import sys


As = np.linspace(0.1, 0.3, 15)
as_str = ",".join([str(a) for a in As])

vs = np.logspace(0, 1, 11)

idx = int(sys.argv[1]) + 1999

if idx%2 == 0:
    pot = "hertz"
else:
    pot = "lj"

files = glob.glob(f"/home/ian/Documents/Data/MD_LV_paper_data/equil/equil_*{pot}*")

assert(len(files) == 1)
f = files[0]

for vscale in vs:

    command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential {pot} \
        --init-config {f} --temp 0.1 --vscale {vscale} --seed {idx} --dt 1e-3 --out-time 1e-2 \
        --dir /home/ian/Documents/Data/MD_LV_paper_data/{pot}_val --time 100 \
        gen-variant --realizations 100 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

    input = command.split()
    subprocess.run(input)