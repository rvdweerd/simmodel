world="Manhattan5x5_DuplicateSetA"
emb=32

etrain="[4,5]"
utrain="[2]"
scen="Train_U$utrain""E$etrain"
echo $scen

echo $emb
for i in {1,3,5}
do
   for j in {"a","b"}
        do
          echo Welcome $i times $j $i"_"$j""$world
        done
done