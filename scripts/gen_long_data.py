import subprocess
import sys
import itertools

num = int(sys.argv[1])

temps = [0.1, 0.04, 0.01, 0.004, 0.001]
time = 100.0


temp = temps[(num-1)%len(temps)]

command = f"md-lv --unwrap --temp {temp} --out-time {time/100} --time {time} --seed {num+100} \
    --init-config /home/igraham/data/md-lv/equil_n-10_l-3.5_dt-1e-3_visc-5_seed-0_phi-0.9489_rA-0.5000_rB-0.7000_vs-1.json \
    --dir /home/igraham/data/md-lv gen-variant --realizations 100000"

input = command.split()
print(input)
subprocess.run(input)