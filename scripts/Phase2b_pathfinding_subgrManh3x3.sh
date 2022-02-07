#!/bin/bash

# ## BATCH
emb=64
numepi=2500 #2500
mem=2000 #2000
tau=100
nstep=2
optim='returns'
train="True"
eval="True"
etrain="0,1,2,3,4,5,6,7,8,9"
etrain_="0123456789"
utrain="1,2,3"
utrain_="123"
scen="Train_U$utrain_""E$etrain_"
numseeds=5
seed0=0
solveselect='solvable'
edgeblock="True"
for nfm in {"NFM_ev_ec_t_um_us",}
do
    for itt in {2,3,4,5,6,7}
    do
        tmux new-session -d -s sub$itt
        tmux send-keys -t "sub$itt" "conda activate rlcourse" Enter
        tmux send-keys -t "sub$itt" "cd ~/testing/sim" Enter
        tmux send-keys -t "sub$itt" "python Phase2b_experiments_Pathfinding_Partial3x3s.py --emb_dim $emb --emb_itT $itt --num_epi $numepi --mem_size $mem --nfm_func $nfm --scenario $scen --optim_target $optim --tau $tau --nstep $nstep --Etrain $etrain --Utrain $utrain --edge_blocking $edgeblock --solve_select $solveselect --train $train --eval $eval --num_seeds $numseeds --seed0 $seed0" Enter
    done
done


# #SINGLE
# sessname="singlerun3"
# emb=64
# itt=4
# nstep=3
# tau=100
# numepi=2500
# mem=2000
# nfm="NFM_ev_ec_t_um_us"
# scen="toptargets-fixed_3U-random-static"
# optim='returns'
# tmux new-session -d -s $sessname
# tmux send-keys -t "$sessname" "conda activate rlcourse" Enter
# tmux send-keys -t "$sessname" "cd ~/testing/sim" Enter
# tmux send-keys -t "$sessname" "python Phase2_experiments_SPath_multiple.py --emb_dim $emb --emb_itT $itt --num_epi $numepi --mem_size $mem --nfm_func $nfm --scenario $scen --optim_target $optim --tau $tau --nstep $nstep" Enter

# # 
# ###################
# tmux new-session -d -s subgr0
# tmux send-keys -t subgr0 "conda activate rlcourse" Enter
# tmux send-keys -t subgr0 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr0 "python Phase2_generate_partial_graphs.py --num_edges 0 --U 1" Enter
# #
# tmux new-session -d -s subgr1
# tmux send-keys -t subgr1 "conda activate rlcourse" Enter
# tmux send-keys -t subgr1 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr1 "python Phase2_generate_partial_graphs.py --num_edges 1 --U 1" Enter
# #
# tmux new-session -d -s subgr2
# tmux send-keys -t subgr2 "conda activate rlcourse" Enter
# tmux send-keys -t subgr2 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr2 "python Phase2_generate_partial_graphs.py --num_edges 2 --U 1" Enter
# #
# tmux new-session -d -s subgr3
# tmux send-keys -t subgr3 "conda activate rlcourse" Enter
# tmux send-keys -t subgr3 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr3 "python Phase2_generate_partial_graphs.py --num_edges 3 --U 1" Enter
# #
# tmux new-session -d -s subgr4
# tmux send-keys -t subgr4 "conda activate rlcourse" Enter
# tmux send-keys -t subgr4 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr4 "python Phase2_generate_partial_graphs.py --num_edges 4 --U 1" Enter
# #
# tmux new-session -d -s subgr5
# tmux send-keys -t subgr5 "conda activate rlcourse" Enter
# tmux send-keys -t subgr5 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr5 "python Phase2_generate_partial_graphs.py --num_edges 5 --U 1" Enter
# #
# tmux new-session -d -s subgr6
# tmux send-keys -t subgr6 "conda activate rlcourse" Enter
# tmux send-keys -t subgr6 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr6 "python Phase2_generate_partial_graphs.py --num_edges 6 --U 1" Enter
# #
# tmux new-session -d -s subgr7
# tmux send-keys -t subgr7 "conda activate rlcourse" Enter
# tmux send-keys -t subgr7 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr7 "python Phase2_generate_partial_graphs.py --num_edges 7 --U 1" Enter
# #
# tmux new-session -d -s subgr8
# tmux send-keys -t subgr8 "conda activate rlcourse" Enter
# tmux send-keys -t subgr8 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr8 "python Phase2_generate_partial_graphs.py --num_edges 8 --U 1" Enter
# #
# tmux new-session -d -s subgr9
# tmux send-keys -t subgr9 "conda activate rlcourse" Enter
# tmux send-keys -t subgr9 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr9 "python Phase2_generate_partial_graphs.py --num_edges 9 --U 1" Enter
# #
# tmux new-session -d -s subgr10
# tmux send-keys -t subgr10 "conda activate rlcourse" Enter
# tmux send-keys -t subgr10 "cd ~/testing/sim" Enter
# tmux send-keys -t subgr10 "python Phase2_generate_partial_graphs.py --num_edges 10 --U 1" Enter
# #####################
# ###################
# # tmux new-session -d -s subgr10
# # tmux send-keys -t subgr10 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr10 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr10 "python Phase2_generate_partial_graphs.py --num_edges 0 --U 2" Enter
# # #
# # tmux new-session -d -s subgr11
# # tmux send-keys -t subgr11 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr11 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr11 "python Phase2_generate_partial_graphs.py --num_edges 1 --U 2" Enter
# # #
# # tmux new-session -d -s subgr12
# # tmux send-keys -t subgr12 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr12 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr12 "python Phase2_generate_partial_graphs.py --num_edges 2 --U 2" Enter
# # #
# # tmux new-session -d -s subgr13
# # tmux send-keys -t subgr13 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr13 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr13 "python Phase2_generate_partial_graphs.py --num_edges 3 --U 2" Enter
# # #
# # tmux new-session -d -s subgr14
# # tmux send-keys -t subgr14 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr14 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr14 "python Phase2_generate_partial_graphs.py --num_edges 4 --U 2" Enter
# # #
# # tmux new-session -d -s subgr15
# # tmux send-keys -t subgr15 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr15 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr15 "python Phase2_generate_partial_graphs.py --num_edges 5 --U 2" Enter
# # #
# # tmux new-session -d -s subgr16
# # tmux send-keys -t subgr16 "conda activate rlcourse" Enter
# # tmux send-keys -t subgr16 "cd ~/testing/sim" Enter
# # tmux send-keys -t subgr16 "python Phase2_generate_partial_graphs.py --num_edges 6 --U 2" Enter
# #
# #####################