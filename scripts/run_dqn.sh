#!/bin/bash

# This script runs eight sessions on the GPU (2MB per session)
###################
world="MetroU3_e17tborder_FixedEscapeInit"
numseeds="5"
for staterep in {"et","etUt","ete0U0","etUte0U0"}
do
    for edgeblock in {"True","False"}
    do
        tmux new-session -d -s DQN_$staterep$edgeblock
        tmux send-keys -t DQN_$staterep$edgeblock "conda activate rlcourse" Enter
        tmux send-keys -t DQN_$staterep$edgeblock "cd ~/testing/sim" Enter
        tmux send-keys -t DQN_$staterep$edgeblock "python Phase1_experiments_DQN.py --world_name " "$world" " --state_repr $staterep --train True --eval True --num_seeds $numseeds --edge_blocking $edgeblock" Enter
    done
done
    #
    
#####################
