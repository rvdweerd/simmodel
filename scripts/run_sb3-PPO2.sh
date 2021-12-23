#!/bin/bash

# This script runs eight sessions on the GPU (2MB per session)
###################
world="Manhattan5x5_VariableEscapeInit"
# tmux new-session -d -s 3_et
# tmux send-keys -t 3_et "conda activate rlcourse" Enter
# tmux send-keys -t 3_et "cd ~/testing/sim" Enter
# tmux send-keys -t 3_et "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr et --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 3_etUt
tmux send-keys -t 3_etUt "conda activate rlcourse" Enter
tmux send-keys -t 3_etUt "cd ~/testing/sim" Enter
tmux send-keys -t 3_etUt "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr etUt --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 3_ete0U0
tmux send-keys -t 3_ete0U0 "conda activate rlcourse" Enter
tmux send-keys -t 3_ete0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 3_ete0U0 "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr ete0U0 --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 3_etUte0U0
tmux send-keys -t 3_etUte0U0 "conda activate rlcourse" Enter
tmux send-keys -t 3_etUte0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 3_etUte0U0 "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr etUte0U0 --TRAIN True --EVAL True" Enter
#####################
