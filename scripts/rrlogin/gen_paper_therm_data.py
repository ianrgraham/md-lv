import glob
import subprocess
import numpy as np
import sys


As = np.linspace(0.1, 0.3, 15)
as_str = ",".join([str(a) for a in As])

temps = np.logspace(-1, -2, 11)

idx = int(sys.argv[1]) + 2999

if idx%2 == 0:
    pot = "hertz"
else:
    pot = "lj"

files = glob.glob(f"/home/igraham/data/MD_LV_paper_data/equil/equil_*{pot}*")

assert(len(files) == 1)
f = files[0]

for temp in temps:

    command = f'/home/igraham/Documents/md-lv/target/release/md-lv --unwrap --potential {pot} \
        --init-config {f} --temp {temp} --seed {idx} --dt 1e-3 --out-time 1e-2 \
        --dir /home/igraham/data/MD_LV_paper_data/{pot}_therm --time 100 \
        gen-variant --realizations 1000 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

    input = command.split()
    subprocess.run(input)