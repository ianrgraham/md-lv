import glob
import subprocess
import numpy as np
import sys
import os


As = np.linspace(0.1, 0.3, 15)
as_str = ",".join([str(a) for a in As])

vs = -np.logspace(0, 1, 11) + 1.0
vs_str = ",".join([str(a) for a in vs])

idx = int(sys.argv[1]) + 13999

if idx%2 == 0:
    pot = "hertz"
    vscale = 1.0
else:
    pot = "lj"
    vscale = 0.1

files = glob.glob(f"/home/igraham/data/MD_LV_pres_data/final_equil/equil_*{pot}*")

assert(len(files) == 1) # should only find one
f = files[0]

out_dir = f"/home/igraham/data/MD_LV_pres_data/{pot}_pred"
os.makedirs(out_dir, exist_ok=True)

command = f'/home/igraham/Documents/md-lv/target/release/md-lv --unwrap --potential {pot} \
    --init-config {f} --vscale {vscale} --temp 0.1 --seed {idx} --dt 1e-3 --out-time 1e-2 \
    --dir {out_dir} --time 1.0 \
    gen-variant --realizations 10000000 --del-var={vs_str} --calc-msd --calc-pos --calc-q={as_str}'

input = command.split()
subprocess.run(input)