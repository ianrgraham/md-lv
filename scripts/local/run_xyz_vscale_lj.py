import glob
import subprocess
import numpy as np

As = np.linspace(0.1, 1.0, 10)

temps = np.linspace(1.0, 11.0, 11) # actually vscale

as_str = ",".join([str(a) for a in As])

# files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/equil_*hertz*")

# for i, f in enumerate(files):
#     print(i)

#     command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --unwrap --potential hertz --time 100.0 \
#         --init-config {f} --seed {100000+i} --out-time 0.1 --dir /home/ian/Documents/Data/MD_LV_paper_data \
#         gen-variant --realizations 100 --del-var=0.0 --calc-msd --calc-pos --calc-q={as_str}'

#     input = command.split()
#     subprocess.run(input)


# files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/inits/equil_*na-5*phi-1.2*lj*")

files = ["/home/ian/Documents/Projects/md-lv/"]

for i, f in enumerate(files):
    print(i)
    for temp in temps:
        command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --potential lj --time 10.0 \
            --init-config {f} --temp 1e-1 --vscale {temp} --seed {200000+i} --dt 1e-4 --out-time 1e-2 \
            --dir /home/ian/Documents/Data/MD_LV_paper_data/xyz_lj'

        input = command.split()
        subprocess.run(input)
    break