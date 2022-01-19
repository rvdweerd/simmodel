#!/bin/bash

# 
###################
tmux new-session -d -s subgr0
tmux send-keys -t subgr0 "conda activate rlcourse" Enter
tmux send-keys -t subgr0 "cd ~/testing/sim" Enter
tmux send-keys -t subgr0 "python Phase2_generate_partial_graphs.py --num_edges 0 --U 1" Enter
#
tmux new-session -d -s subgr1
tmux send-keys -t subgr1 "conda activate rlcourse" Enter
tmux send-keys -t subgr1 "cd ~/testing/sim" Enter
tmux send-keys -t subgr1 "python Phase2_generate_partial_graphs.py --num_edges 1 --U 1" Enter
#
tmux new-session -d -s subgr2
tmux send-keys -t subgr2 "conda activate rlcourse" Enter
tmux send-keys -t subgr2 "cd ~/testing/sim" Enter
tmux send-keys -t subgr2 "python Phase2_generate_partial_graphs.py --num_edges 2 --U 1" Enter
#
tmux new-session -d -s subgr3
tmux send-keys -t subgr3 "conda activate rlcourse" Enter
tmux send-keys -t subgr3 "cd ~/testing/sim" Enter
tmux send-keys -t subgr3 "python Phase2_generate_partial_graphs.py --num_edges 3 --U 1" Enter
#
tmux new-session -d -s subgr4
tmux send-keys -t subgr4 "conda activate rlcourse" Enter
tmux send-keys -t subgr4 "cd ~/testing/sim" Enter
tmux send-keys -t subgr4 "python Phase2_generate_partial_graphs.py --num_edges 4 --U 1" Enter
#
tmux new-session -d -s subgr5
tmux send-keys -t subgr5 "conda activate rlcourse" Enter
tmux send-keys -t subgr5 "cd ~/testing/sim" Enter
tmux send-keys -t subgr5 "python Phase2_generate_partial_graphs.py --num_edges 5 --U 1" Enter
#
tmux new-session -d -s subgr6
tmux send-keys -t subgr6 "conda activate rlcourse" Enter
tmux send-keys -t subgr6 "cd ~/testing/sim" Enter
tmux send-keys -t subgr6 "python Phase2_generate_partial_graphs.py --num_edges 6 --U 1" Enter
#
#####################
###################
tmux new-session -d -s subgr10
tmux send-keys -t subgr10 "conda activate rlcourse" Enter
tmux send-keys -t subgr10 "cd ~/testing/sim" Enter
tmux send-keys -t subgr10 "python Phase2_generate_partial_graphs.py --num_edges 0 --U 2" Enter
#
tmux new-session -d -s subgr11
tmux send-keys -t subgr11 "conda activate rlcourse" Enter
tmux send-keys -t subgr11 "cd ~/testing/sim" Enter
tmux send-keys -t subgr11 "python Phase2_generate_partial_graphs.py --num_edges 1 --U 2" Enter
#
tmux new-session -d -s subgr12
tmux send-keys -t subgr12 "conda activate rlcourse" Enter
tmux send-keys -t subgr12 "cd ~/testing/sim" Enter
tmux send-keys -t subgr12 "python Phase2_generate_partial_graphs.py --num_edges 2 --U 2" Enter
#
tmux new-session -d -s subgr13
tmux send-keys -t subgr13 "conda activate rlcourse" Enter
tmux send-keys -t subgr13 "cd ~/testing/sim" Enter
tmux send-keys -t subgr13 "python Phase2_generate_partial_graphs.py --num_edges 3 --U 2" Enter
#
tmux new-session -d -s subgr14
tmux send-keys -t subgr14 "conda activate rlcourse" Enter
tmux send-keys -t subgr14 "cd ~/testing/sim" Enter
tmux send-keys -t subgr14 "python Phase2_generate_partial_graphs.py --num_edges 4 --U 2" Enter
#
tmux new-session -d -s subgr15
tmux send-keys -t subgr15 "conda activate rlcourse" Enter
tmux send-keys -t subgr15 "cd ~/testing/sim" Enter
tmux send-keys -t subgr15 "python Phase2_generate_partial_graphs.py --num_edges 5 --U 2" Enter
#
tmux new-session -d -s subgr16
tmux send-keys -t subgr16 "conda activate rlcourse" Enter
tmux send-keys -t subgr16 "cd ~/testing/sim" Enter
tmux send-keys -t subgr16 "python Phase2_generate_partial_graphs.py --num_edges 6 --U 2" Enter
#
#####################