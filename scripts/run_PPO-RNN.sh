#!/bin/bash

#tmux kill-server

# This script runs eight sessions on the GPU (2MB per session)
###################
# world="MetroU3_e17t31_FixedEscapeInit"
# tmux new-session -d -s 1_et
# tmux send-keys -t 1_et "conda activate rlcourse" Enter
# tmux send-keys -t 1_et "cd ~/testing/sim" Enter
# tmux send-keys -t 1_et "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr et --train True --eval True" Enter

# tmux new-session -d -s 1_etUt
# tmux send-keys -t 1_etUt "conda activate rlcourse" Enter
# tmux send-keys -t 1_etUt "cd ~/testing/sim" Enter
# tmux send-keys -t 1_etUt "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr etUt --train True --eval True" Enter

# tmux new-session -d -s 1_ete0U0
# tmux send-keys -t 1_ete0U0 "conda activate rlcourse" Enter
# tmux send-keys -t 1_ete0U0 "cd ~/testing/sim" Enter
# tmux send-keys -t 1_ete0U0 "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr ete0U0 --train True --eval True" Enter
# #
# tmux new-session -d -s 1_etUte0U0
# tmux send-keys -t 1_etUte0U0 "conda activate rlcourse" Enter
# tmux send-keys -t 1_etUte0U0 "cd ~/testing/sim" Enter
# tmux send-keys -t 1_etUte0U0 "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr etUte0U0 --train True --eval True" Enter
#####################
###################
world="MetroU3_e17t31_FixedEscapeInit"
tmux new-session -d -s 2_et
tmux send-keys -t 2_et "conda activate rlcourse" Enter
tmux send-keys -t 2_et "cd ~/testing/sim" Enter
tmux send-keys -t 2_et "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr et --train True --eval True" Enter

tmux new-session -d -s 2_etUt
tmux send-keys -t 2_etUt "conda activate rlcourse" Enter
tmux send-keys -t 2_etUt "cd ~/testing/sim" Enter
tmux send-keys -t 2_etUt "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr etUt --train True --eval True" Enter
#
tmux new-session -d -s 2_ete0U0
tmux send-keys -t 2_ete0U0 "conda activate rlcourse" Enter
tmux send-keys -t 2_ete0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 2_ete0U0 "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr ete0U0 --train True --eval True" Enter
#
tmux new-session -d -s 2_etUte0U0
tmux send-keys -t 2_etUte0U0 "conda activate rlcourse" Enter
tmux send-keys -t 2_etUte0U0 "cd ~/testing/sim" Enter
tmux send-keys -t 2_etUte0U0 "python Phase1_experiments_PPO-RNN.py --world_name " "$world" " --state_repr etUte0U0 --train True --eval True" Enter
# #####################