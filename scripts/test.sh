world="Manhattan5x5_DuplicateSetA"
emb=32
echo $emb
for i in {1,3,5}
do
   for j in {"a","b"}
        do
          echo Welcome $i times $j $i"_"$j""$world
        done
done