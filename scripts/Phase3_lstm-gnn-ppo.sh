#!/bin/bash

#train_on="MixAll33"
#train_on="NWB_AMS_mixed_obs"
#train_on="TEST"
#train_on="NWB_AMS"
train_on="HeurCRE"
#train_on="M5x5Fixed"
#train_on="M5x5F_mixed_obs"
#train_on="MemTask-U1"
batch_size=2
#obs_mask="mix"
obs_mask="None"
obs_rate=0.0
emb_dim=64
#lstm_hdim=64
lstm_hdim=64
lstm_layers=1
emb_iterT=5
nfm_func="NFM_ev_ec_t_dt_at_um_us"
#nfm_func="NFM_ev_ec_t_dt_at_ustack"
qnet="gat2"
critic='q'
train="False"
eval="False"
test="False"
test_heur="True"
num_seeds=1
#seed0=0
demoruns="False"
num_step=25001
type_obs_wrap="BasicDict"
parallel_rollouts=4
idn="heur"
id="ams-u15"
eval_deter="True"
eval_rate=-0.5
#idn="AMSmix-TEST"
#id="EMB64"
for lstm_type in {"None",}
do
    for seed0 in {0,}
    do
        tmux new-session -d -s "${idn}-${id}"
        tmux send-keys -t "${idn}-${id}" "conda activate rlcourse-sb3c" Enter
        tmux send-keys -t "${idn}-${id}" "cd ~/testing/sim" Enter
        tmux send-keys -t "${idn}-${id}" "python Phase3_lstm-gnn-ppo_simp.py --train_on $train_on --batch_size $batch_size --obs_mask $obs_mask --obs_rate $obs_rate --emb_dim $emb_dim --lstm_type $lstm_type --lstm_hdim $lstm_hdim --lstm_layers $lstm_layers --emb_iterT $emb_iterT --nfm_func $nfm_func --qnet $qnet --train $train --eval $eval --test $test --num_seeds $num_seeds --seed0 $seed0 --demoruns $demoruns --parallel_rollouts $parallel_rollouts --critic $critic --num_step $num_step --test_heur $test_heur --type_obs_wrap $type_obs_wrap --eval_deter $eval_deter --eval_rate $eval_rate" Enter
    done
done
