#!/bin/bash

# This script runs eight sessions on the GPU (2MB per session)
###################
world="all"
tmux new-session -d -s DRQN_et
tmux send-keys -t DRQN_et "conda activate rlcourse" Enter
tmux send-keys -t DRQN_et "cd ~/testing/sim" Enter
tmux send-keys -t DRQN_et "python Phase1_experiments_DRQN.py --world_name " "$world" " --state_repr et --train True --eval True --num_seeds 5" Enter
#
tmux new-session -d -s DRQN_etUt
tmux send-keys -t DRQN_etUt "conda activate rlcourse" Enter
tmux send-keys -t DRQN_etUt "cd ~/testing/sim" Enter
tmux send-keys -t DRQN_etUt "python Phase1_experiments_DRQN.py --world_name " "$world" " --state_repr etUt --train True --eval True --num_seeds 5" Enter
# #
tmux new-session -d -s DRQN_ete0U0
tmux send-keys -t DRQN_ete0U0 "conda activate rlcourse" Enter
tmux send-keys -t DRQN_ete0U0 "cd ~/testing/sim" Enter
tmux send-keys -t DRQN_ete0U0 "python Phase1_experiments_DRQN.py --world_name " "$world" " --state_repr ete0U0 --train True --eval True --num_seeds 5" Enter
# #
tmux new-session -d -s DRQN_etUte0U0
tmux send-keys -t DRQN_etUte0U0 "conda activate rlcourse" Enter
tmux send-keys -t DRQN_etUte0U0 "cd ~/testing/sim" Enter
tmux send-keys -t DRQN_etUte0U0 "python Phase1_experiments_DRQN.py --world_name " "$world" " --state_repr etUte0U0 --train True --eval True --num_seeds 5" Enter
#####################
