#!/bin/bash

# 
###################
tmux new-session -d -s subgr0
tmux send-keys -t subgr0 "conda activate rlcourse" Enter
tmux send-keys -t subgr0 "cd ~/testing/sim" Enter
tmux send-keys -t subgr0 "python Phase2_generate_partial_graphs.py --num_edges 0" Enter
#
tmux new-session -d -s subgr1
tmux send-keys -t subgr1 "conda activate rlcourse" Enter
tmux send-keys -t subgr1 "cd ~/testing/sim" Enter
tmux send-keys -t subgr1 "python Phase2_generate_partial_graphs.py --num_edges 1" Enter
#
tmux new-session -d -s subgr2
tmux send-keys -t subgr2 "conda activate rlcourse" Enter
tmux send-keys -t subgr2 "cd ~/testing/sim" Enter
tmux send-keys -t subgr2 "python Phase2_generate_partial_graphs.py --num_edges 2" Enter
#
tmux new-session -d -s subgr3
tmux send-keys -t subgr3 "conda activate rlcourse" Enter
tmux send-keys -t subgr3 "cd ~/testing/sim" Enter
tmux send-keys -t subgr3 "python Phase2_generate_partial_graphs.py --num_edges 3" Enter
#
tmux new-session -d -s subgr4
tmux send-keys -t subgr4 "conda activate rlcourse" Enter
tmux send-keys -t subgr4 "cd ~/testing/sim" Enter
tmux send-keys -t subgr4 "python Phase2_generate_partial_graphs.py --num_edges 4" Enter
#
tmux new-session -d -s subgr5
tmux send-keys -t subgr5 "conda activate rlcourse" Enter
tmux send-keys -t subgr5 "cd ~/testing/sim" Enter
tmux send-keys -t subgr5 "python Phase2_generate_partial_graphs.py --num_edges 5" Enter
#
tmux new-session -d -s subgr6
tmux send-keys -t subgr6 "conda activate rlcourse" Enter
tmux send-keys -t subgr6 "cd ~/testing/sim" Enter
tmux send-keys -t subgr6 "python Phase2_generate_partial_graphs.py --num_edges 6" Enter
#
#####################
