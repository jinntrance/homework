#!/usr/bin/env bash
python3 train_pg.py CartPole-v1 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna
python3 train_pg.py CartPole-v1 -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna
python3 train_pg.py CartPole-v1 -n 100 -b 1000 -e 5 -rtg --exp_name sb_rtg_na
python3 train_pg.py CartPole-v1 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna
python3 train_pg.py CartPole-v1 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna
python3 train_pg.py CartPole-v1 -n 100 -b 5000 -e 5 -rtg --exp_name lb_rtg_na

for i in data/*; do
    python3 plot.py $i;
done