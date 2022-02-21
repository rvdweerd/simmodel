#!/bin/bash

# ## BATCH
train="False"
eval="True"
nr="1"
for run in {"TestHeur",}
#"RunB","RunC","RunD","RunE","train_on_metro"}
do
        tmux new-session -d -s $nr$run
        tmux send-keys -t "$nr$run" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "$nr$run" "cd ~/testing/sim" Enter
        tmux send-keys -t "$nr$run" "python Phase2d_gnn-ppo_sb3.py --run_name $run --train $train --eval $eval" Enter
done

