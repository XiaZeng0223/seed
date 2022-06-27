
#reproduce the seed experiments on any of the implemented datasets
#specify the name of the task here; available options are 'bfever' 'fever' 'scifact'
task='scifact'

for model in bert-base-uncased bert-large-uncased roberta-base roberta-large bert-base-nli-mean-tokens
  do
  for seed in 123 124 125 126 127 128 129 130 131 132
    do
    for m in 2 4 6 8 10 20 30 40 50 100
      do
      echo $m; echo $model; echo $seed
      if [[ "$task" == "bfever" ]]; then
        time python seed_fever_binary.py --m $m --model $model --seed $seed --abs True
      elif [[ "$task" == "fever" ]]; then
        time python seed_fever.py --m $m --model $model --seed $seed --abs True
      elif [[ "$task" == "scifact" ]]; then
        time python seed_scifact.py --m $m --model $model --seed $seed --abs True
      fi
      done
    done
  done

