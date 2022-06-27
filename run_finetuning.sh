

#reproduce the seed experiments on any of the implemented datasets
#specify the name of the task here; available options are 'bfever' 'fever' 'scifact'
task='fever'

for model in roberta-base bert-base-uncased bert-large-uncased  roberta-large bert-base-nli-mean-tokens
  do
  for seed in 123 124 125 126 127 128 129 130 131 132
    do
    for m in 2 4 6 8 10 20 30 40 50 100
      do
        for run in  1 2 3 4 5
        do
          echo $m; echo $model; echo $seed; echo 'index of run' $run
          if [ "$model" == "roberta-base" -o "$model" == "roberta-large" ]; then
            time python finetune.py --dataset $task --m $m --model $model --model_base roberta --seed $seed --lr-base 5e-6 --lr-linear 5e-6
          else
            time python finetune.py --dataset $task --m $m --model $model --model_base bert --seed $seed --lr-base 5e-6 --lr-linear 5e-6
          fi
        done
      done
    done
  done
