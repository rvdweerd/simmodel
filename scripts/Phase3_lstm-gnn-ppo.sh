#!/bin/bash

#tmux kill-server

#train_on="MixAll33"
train_on="NWB_AMS+Mix"
batch_size=10
mask_type="None"
mask_rate=0
emb_dim=64
lstm_hdim=56
lstm_layers=1
emb_iterT=5
nfm_func="NFM_ev_ec_t_dt_at_um_us"
qnet="gat2"
train="True"
eval="False"
test="False"
num_seeds=5
seed0=0
demoruns="False"
parallel_rollouts=1

#"separate-noncat","shared-noncat"}
for lstm_type in {"shared-concat",}
do
    tmux new-session -d -s "${lstm_type}"
    tmux send-keys -t "${lstm_type}" "conda activate rlcourse-sb3c" Enter
    tmux send-keys -t "${lstm_type}" "cd ~/testing/sim" Enter
    tmux send-keys -t "${lstm_type}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --mask_type $mask_type --mask_rate $mask_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts" Enter
done

#lstm_type="separate-noncat"
#lstm_type2="None"
#tmux send-keys -t "${lstm_type}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --mask_type $mask_type --mask_rate $mask_rate --emb_dim $emb_dim --lstm_type $lstm_type2 --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns" Enter
