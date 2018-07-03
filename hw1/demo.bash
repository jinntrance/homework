#!/bin/bash
set -eux
for e in Hopper Ant HalfCheetah Humanoid Reacher Walker2d
do
    python3 run_expert.py experts/$e-v1.pkl "${e}-v2" --render --num_rollouts=10
done
