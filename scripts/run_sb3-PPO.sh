#!/bin/bash

# This script runs eight sessions on the GPU (2MB per session)
###################
world="Manhattan5x5_DuplicateSetA"
tmux new-session -d -s 1_et
tmux send-keys -t 1_et "conda activate rlcourse" Enter
tmux send-keys -t 1_et "cd ~/testing/sim" Enter
tmux send-keys -t 1_et "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr et --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 1_etUt
tmux send-keys -t 1_etUt "conda activate rlcourse" Enter
tmux send-keys -t 1_etUt "cd ~/testing/sim" Enter
tmux send-keys -t 1_etUt "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr etUt --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 1_ete0U0
tmux send-keys -t 1_ete0U0 "conda activate rlcourse" Enter
tmux send-keys -t 1_ete0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 1_ete0U0 "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr ete0U0 --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 1_etUte0U0
tmux send-keys -t 1_etUte0U0 "conda activate rlcourse" Enter
tmux send-keys -t 1_etUte0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 1_etUte0U0 "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr etUte0U0 --TRAIN True --EVAL True" Enter
#####################

###################
world="Manhattan5x5_DuplicateSetB"
tmux new-session -d -s 2_et
tmux send-keys -t 2_et "conda activate rlcourse" Enter
tmux send-keys -t 2_et "cd ~/testing/sim" Enter
tmux send-keys -t 2_et "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr et --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 2_etUt
tmux send-keys -t 2_etUt "conda activate rlcourse" Enter
tmux send-keys -t 2_etUt "cd ~/testing/sim" Enter
tmux send-keys -t 2_etUt "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr etUt --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 2_ete0U0
tmux send-keys -t 2_ete0U0 "conda activate rlcourse" Enter
tmux send-keys -t 2_ete0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 2_ete0U0 "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr ete0U0 --TRAIN True --EVAL True" Enter
#
tmux new-session -d -s 2_etUte0U0
tmux send-keys -t 2_etUte0U0 "conda activate rlcourse" Enter
tmux send-keys -t 2_etUte0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 2_etUte0U0 "python Phase1_experiments_sb3-PPO.py --world_name " "$world" " --state_repr etUte0U0 --TRAIN True --EVAL True" Enter
#####################