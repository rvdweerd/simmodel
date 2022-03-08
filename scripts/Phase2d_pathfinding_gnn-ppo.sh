#!/bin/bash

# ## BATCH
train="True"
eval="False"
test="True"
nr="ppo10"
for run in {"SuperSet_noU",}
#"RunB","RunC","RunD","RunE","train_on_metro"}
do
        tmux new-session -d -s $nr$run
        tmux send-keys -t "$nr$run" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "$nr$run" "cd ~/testing/sim" Enter
        tmux send-keys -t "$nr$run" "python Phase2d_gnn-ppo_sb3.py --run_name $run --train $train --eval $eval --test $test" Enter
done

