# world="Manhattan5x5_DuplicateSetA"
# emb=32

# etrain="[4,5]"
# utrain="[2]"
# scen="Train_U$utrain""E$etrain"
# echo $scen

# echo $emb
# for i in {1,3,5}
# do
#    for j in {"a","b"}
#         do
#           echo Welcome $i times $j $i"_"$j""$world
#         done
# done
tmux new-session -d -s test
tmux send-keys -t test "conda activate rlcourse-sb3c" Enter
tmux send-keys -t test "cd ~/testing/sim" Enter
tmux send-keys -t test "python test.py" Enter
tmux send-keys -t test "python test.py" Enter
tmux send-keys -t test "python test.py" Enter
tmux send-keys -t test "python test.py" Enter
tmux send-keys -t test "python test.py" Enter

