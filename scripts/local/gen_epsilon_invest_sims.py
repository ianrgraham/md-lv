import glob
import subprocess
import numpy as np
import sys

As = np.linspace(0.1, 0.3, 15)
as_str = ",".join([str(a) for a in As])

temps = np.logspace(0, 1, 11)

files = glob.glob("/home/ian/Documents/Data/MD_LV_investigate/equil/equil_*na-5*seed-1*hertz*")

idx = 500

for i, f in enumerate(files):
    print(i)
    for temp in temps:
        print(temp)

        command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential hertz --time 100.0 \
            --init-config {f} --vscale {temp} --seed {idx} --dt 1e-3 --out-time 1e-2 \
            --dir /home/ian/Documents/Data/MD_LV_investigate/epsilon_hertz \
            gen-variant --realizations 100 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

        input = command.split()
        subprocess.run(input)
        idx += 1
    break

files = glob.glob("/home/ian/Documents/Data/MD_LV_investigate/equil/equil_*lj*")

for i, f in enumerate(files):
    print(i)
    for temp in temps:
        print(temp)

        command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential lj --time 100.0 \
            --init-config {f} --vscale {temp} --seed {idx} --dt 1e-3 --out-time 1e-2 \
            --dir /home/ian/Documents/Data/MD_LV_investigate/epsilon_lj \
            gen-variant --realizations 100 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

        input = command.split()
        subprocess.run(input)
        idx += 1
    break