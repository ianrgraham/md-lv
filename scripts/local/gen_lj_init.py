import glob
import subprocess

files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/equil_*hertz*")
# print(files)

for i, f in enumerate(files):
    print(i)

    command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --potential lj --time 1000000 \
        --init-config {f} --seed {10030+i} --dir /home/ian/Documents/Data/MD_LV_paper_data equil-gd '

    input = command.split()
    subprocess.run(input)