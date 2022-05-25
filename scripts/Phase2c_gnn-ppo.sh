#!/bin/bash

# ## BATCH
#train_on="M3M5Mix"
#train_on="NWB_AMS"
#train_on="MixM5Met_T"
train_on="M3_012456"
#train_on="MixAll33"
#train_on="TEST"
emb_dim=64
itt=5
nfm="NFM_ev_ec_t_dt_at_um_us"
#qnet="s2v"
norm_agg="True"
optim='returns'
nstep=200000 #200000
train="True"
eval="False"
test="False"
numseeds=1
#seed0=1
solveselect='solvable'
edgeblock="True"
max_nodes=9
demoruns="False"
pursuit="Uon"

for seed0 in {"0",}
#"NFM_ec_t","NFM_ec_dtscaled"}
do
    for qnet in {"gat2",}
    do
        tmux new-session -d -s "${qnet}-${seed0}"
        tmux send-keys -t "${qnet}-${seed0}" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "${qnet}-${seed0}" "cd ~/testing/sim" Enter
        tmux send-keys -t "${qnet}-${seed0}" "python Phase2c_gnn-ppo.py --train_on $train_on --emb_dim $emb_dim --emb_itT $itt --nfm_func $nfm --qnet $qnet --norm_agg $norm_agg --optim_target $optim --num_step $nstep --train $train --eval $eval --test $test --num_seeds $numseeds --seed0 $seed0 --solve_select $solveselect --edge_blocking $edgeblock --max_nodes $max_nodes --demoruns $demoruns --pursuit $pursuit" Enter
    done
done