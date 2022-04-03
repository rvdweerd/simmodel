#!/bin/bash

#train_on="MixAll33"
#train_on="NWB_AMS+Mix"
train_on="M5x5Fixed"
batch_size=48
#obs_mask="None"
#obs_rate=0
obs_mask="prob_per_u"
obs_rate=0.75
emb_dim=64
#lstm_hdim=64
#lstm_hdim=24
lstm_layers=1
emb_iterT=5
#nfm_func="NFM_ev_ec_t_dt_at_um_us"
nfm_func="NFM_ev_ec_t_dt_at_ustack"
qnet="gat2"
train="True"
eval="False"
test="False"
num_seeds=5
seed0=0
demoruns="False"
parallel_rollouts=2
rollout_steps=150

# RUNNING JOBS IN SERIES (no loops)
# terminalname="seriesrun"
# tmux new-session -d -s "${terminalname}"
# tmux send-keys -t "${terminalname}" "conda activate rlcourse-sb3c" Enter
# tmux send-keys -t "${terminalname}" "cd ~/testing/sim" Enter

# lstm_type="None"
# tmux send-keys -t "${terminalname}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter
# lstm_type="shared-concat"
# tmux send-keys -t "${terminalname}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter
# lstm_type="separate-noncat"
# tmux send-keys -t "${terminalname}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter

#RUNNING JOBS IN PARALLEL
for lstm_type in {"FE",}
#"shared-noncat","separate-noncat","shared-concat","None"}
do
   for lstm_hdim in {24,64}
   do
      tmux new-session -d -s "${lstm_type}p2${lstm_hdim}"
      tmux send-keys -t "${lstm_type}p2${lstm_hdim}" "conda activate rlcourse-sb3c" Enter
      tmux send-keys -t "${lstm_type}p2${lstm_hdim}" "cd ~/testing/sim" Enter
      tmux send-keys -t "${lstm_type}p2${lstm_hdim}" "python Phase3_lstm-gnn-ppo.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --rollout_steps $rollout_steps" Enter
   done
done
