import glob
import subprocess

files = glob.glob("/home/ian/Documents/Data/MD_LV_paper_data/equil/equil_n-25_na-9_l-4.639065309883615_dt-1e-3_visc-5_seed-1000_phi-1.1617_pot-hertz_rs-1_vs-1.json")
print(files)

for i, f in enumerate(files):
    print(i)

    command = f'/home/ian/Documents/Projects/md-lv/target/release/md-lv --potential lj --time 1 \
        --init-config {f} --dt 1e-3 --temp 1e-2 --seed {9900+i} --dir /home/ian/Documents/Data/MD_LV_paper_data/equil equil-gd'

    input = command.split()
    subprocess.run(input)