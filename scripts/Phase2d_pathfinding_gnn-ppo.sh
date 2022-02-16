#!/bin/bash

# ## BATCH
for run in {"RunB","RunC","RunD","RunE","train_on_metro"}
do
        tmux new-session -d -s $run
        tmux send-keys -t "$run" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "$run" "cd ~/testing/sim" Enter
        tmux send-keys -t "$run" "python Phase2d_gnn-ppo_sb3.py --run_name $run" Enter
done

