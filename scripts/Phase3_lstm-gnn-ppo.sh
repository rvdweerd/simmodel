#!/bin/bash

#train_on="MixAll33"
#train_on="NWB_AMS"
#train_on="M5x5Fixed"
#train_on="M3x5Mix"
train_on="MemTask-U1"
batch_size=2
obs_mask="freq"
obs_rate=.2
#obs_mask="prob_per_u"
#obs_rate=.5
emb_dim=24
#lstm_hdim=64
lstm_hdim=24
lstm_layers=1
emb_iterT=5
#nfm_func="NFM_ev_ec_t_dt_at_um_us"
nfm_func="NFM_ev_ec_t_dt_at_ustack"
qnet="gat2"
#critic='v'
train="True"
eval="False"
test="False"
num_seeds=10
seed0=0
demoruns="False"
rollout_steps=150
num_step=1000

parallel_rollouts=4
for lstm_type in {"None",}
do
    for critic in {"q","v"}
    do
        tmux new-session -d -s "${lstm_type}-stck-${critic}"
        tmux send-keys -t "${lstm_type}-stck-${critic}" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "${lstm_type}-stck-${critic}" "cd ~/testing/sim" Enter
        tmux send-keys -t "${lstm_type}-stck-${critic}" "python Phase3_lstm-gnn-ppo_simp.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps --critic $critic --num_step $num_step" Enter
    done
done