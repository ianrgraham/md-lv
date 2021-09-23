import glob
import subprocess
import numpy as np

As = np.linspace(0.1, 1.0, 10)

temps = np.logspace(-3, -2, 10)

as_str = ",".join([str(a) for a in As])

# files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/equil_*hertz*")

# for i, f in enumerate(files):
#     print(i)

#     command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential hertz --time 100.0 \
#         --init-config {f} --seed {100000+i} --out-time 0.1 --dir /home/ian/Documents/Data/MD_LV_paper_data \
#         gen-variant --realizations 100 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

#     input = command.split()
#     subprocess.run(input)


files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/inits/equil_*lj*")

for i, f in enumerate(files):
    print(i)
    command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --potential hertz --time 10000.0 \
        --init-config {f} --temp 1e-2 --seed {200000+i} --dt 1e-3 --out-time 1e-1 \
        --dir /home/ian/Documents/Data/MD_LV_paper_data/small_find_glass_transition'

    input = command.split()
    subprocess.run(input)
    break