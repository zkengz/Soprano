#!/bin/bash

# tau comparison

# for tau in {0,0.5,2}; do
#     for s in {0..4}; do
#         /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 4 --maxbets 1 --random_seed $s --algo MCCFR --tau $tau --n_traj 10 --lr 5e-3 --n_iter 50000 &
#         /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 4 --maxbets 3 --random_seed $s --algo MCCFR --tau $tau --n_traj 10 --lr 5e-3 --n_iter 50000 &
#         /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 13 --maxbets 1 --random_seed $s --algo MCCFR --tau $tau --n_traj 10 --lr 5e-3 --n_iter 50000 &
#     done
# done

# n_traj comparison

# 2,4,6,10,20,40

# for algo in {"MCCFR-history",}; do
#     for ntraj in {40,}; do
#         for s in {0..4}; do
#             /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 4 --maxbets 1 --random_seed $s --algo $algo --tau 0.5 --n_traj $ntraj --lr 5e-3 --n_iter 50000 &
#             /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 4 --maxbets 3 --random_seed $s --algo $algo --tau 0.5 --n_traj $ntraj --lr 5e-3 --n_iter 50000 &
#             /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 13 --maxbets 1 --random_seed $s --algo $algo --tau 0.5 --n_traj $ntraj --lr 5e-3 --n_iter 50000 &
#         done
#     done
# done

# algo comparison

for algo in {"MCCFR-infoset",}; do
    for s in {0..4}; do
        /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 4 --maxbets 1 --random_seed $s --algo $algo --tau 0.5 --n_traj 20 --lr 5e-3 --n_iter 50000 &
        /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 4 --maxbets 3 --random_seed $s --algo $algo --tau 0.5 --n_traj 20 --lr 5e-3 --n_iter 50000 &
        /home/zengzekeng/anaconda3/envs/alterTPG/bin/python run_kuhn_2v1_so.py --rank 13 --maxbets 1 --random_seed $s --algo $algo --tau 0.5 --n_traj 20 --lr 5e-3 --n_iter 50000 &
    done
done