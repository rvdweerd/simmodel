#!/bin/bash

#train_on="MixAll33"
train_on="NWB_AMS"
#train_on="M5x5Fixed"
#train_on="M3x5Mix"
batch_size=2
obs_mask="None"
obs_rate=1.0
#obs_mask="prob_per_u"
#obs_rate=.5
emb_dim=64
#lstm_hdim=64
#lstm_type="Dual"
#lstm_type="Dual"
lstm_hdim=64
lstm_layers=1
emb_iterT=5
#nfm_func="NFM_ev_ec_t_dt_at_um_us"
#nfm_func="NFM_ev_ec_t_dt_at_ustack"
qnet="gat2"
train="False"
eval="False"
test="True"
num_seeds=20
seed0=0
demoruns="False"
rollout_steps=150

# RUNNING JOBS IN SERIES (no loops)
# terminalname="seriesrun"
# tmux new-session -d -s "${terminalname}"
# tmux send-keys -t "${terminalname}" "conda activate rlcourse-sb3c" Enter
# tmux send-keys -t "${terminalname}" "cd ~/testing/sim" Enter

# tmux send-keys -t "${terminalname}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter
# lstm_type="shared-concat"
# tmux send-keys -t "${terminalname}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter
# lstm_type="separate-noncat"
# tmux send-keys -t "${terminalname}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter

#RUNNING JOBS IN PARALLEL
#for obs_mask in {"prob_per_u","None"}
#"shared-noncat","separate-noncat","shared-concat","None"}
#   for lstm_hdim in {24,64}
#for lstm_type in {"FE","None"}
#do
#"NFM_ev_ec_t_dt_at_ustack"}
#id="test1"
nfm_func="NFM_ev_ec_t_dt_at_um_us"
#lstm_type="Dual"
parallel_rollouts=4
for lstm_type in {"Dual",}
do
    tmux new-session -d -s "${lstm_type}t-${seed0}"
    tmux send-keys -t "${lstm_type}t-${seed0}" "conda activate rlcourse-sb3c" Enter
    tmux send-keys -t "${lstm_type}t-${seed0}" "cd ~/testing/sim" Enter
    tmux send-keys -t "${lstm_type}t-${seed0}" "python Phase3_lstm-gnn-ppo_simp.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter
done
