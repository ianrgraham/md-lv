import glob
import subprocess
import numpy as np
import sys

As = np.linspace(0.1, 0.3, 15)
as_str = ",".join([str(a) for a in As])

temps = np.logspace(-1, 0, 10, endpoint=False)/10

files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/equil/equil_n-10_na-5_l-3.1622776601683795_dt-1e-3_visc-5_seed-30_phi-1.0000_pot-hertz_rs-1_vs-1.json")

idx = 3000

# for i, f in enumerate(files):
#     print(i)
#     for temp in temps:
#         print(temp)

#         command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential hertz --time 100.0 \
#             --init-config {f} --vscale {temp} --seed {idx} --dt 1e-3 --out-time 1e-2 \
#             --dir /home/ian/Documents/Data/MD_LV_investigate/epsilon_hertz_new \
#             gen-variant --realizations 100 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

#         input = command.split()
#         subprocess.run(input)
#         idx += 1
#     break

files = glob.glob("/home/ian/Documents/Projects/md-lv/equil_n-10_na-4_l-3.015113445777636_dt-1e-3_visc-5_seed-203_phi-1.1000_pot-lj_rs-1_vs-0.01.json")

for i, f in enumerate(files):
    print(i)
    for temp in temps:
        print(temp)

        command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential lj --time 10.0 \
            --init-config {f} --vscale {temp} --seed {idx} --dt 1e-3 --out-time 1e-2 \
            --dir /home/ian/Documents/Data/MD_LV_investigate/epsilon_lj_new \
            gen-variant --realizations 1000 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

        input = command.split()
        subprocess.run(input)
        idx += 1
    break