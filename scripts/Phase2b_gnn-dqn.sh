#!/bin/bash

# ## BATCH
emb=64
numepi=25000 #2500
mem=5000 #2000
tau=100
nstep=2
optim='returns'
train="False"
eval="False"
test="True"
etrain="0,1,2,3,4,5,6,7,8,9"
etrain_="0123456789"
utrain="1,2,3"
utrain_="123"
#qnet="s2v"
train_on="M3M5Mix"
#train_on="NWB_AMS"
#train_on="MixAll33"
pursuit="Uon"
solveselect='solvable'
edgeblock="True"
nfm="NFM_ev_ec_t_dt_at_um_us"
itt=5
max_nodes=25
demoruns="False"
#seed0=1
numseeds=5
norm_agg="True"

for seed0 in {"1",}
#"2","3","4","5"}
#"NFM_ec_t","NFM_ec_dtscaled"}
do
    for qnet in {"s2v",}
    do
        tmux new-session -d -s "${qnet}test1-${seed0}"
        tmux send-keys -t "${qnet}test1-${seed0}" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "${qnet}test1-${seed0}" "cd ~/testing/sim" Enter
        tmux send-keys -t "${qnet}test1-${seed0}" "python Phase2b_gnn-dqn.py --emb_dim $emb --emb_itT $itt --num_epi $numepi --mem_size $mem --nfm_func $nfm --qnet $qnet --norm_agg $norm_agg --train_on $train_on --max_nodes $max_nodes --pursuit $pursuit --optim_target $optim --tau $tau --nstep $nstep --Etrain $etrain --Utrain $utrain --edge_blocking $edgeblock --solve_select $solveselect --train $train --eval $eval --test $test --num_seeds $numseeds --seed0 $seed0 --demoruns $demoruns" Enter
    done
done