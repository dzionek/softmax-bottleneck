##!/bin/bash
MODELS=('softmax' 'mos' 'sigsoftmax' 'moss' 'plif')
DS=('128' '9' '8' '7' '6' '5' '4' '3' '2' '1')

# Run MNIST
for model in "${MODELS[@]}"
do
  for d in "${DS[@]}"
  do
    NAME="${model}_${d}"
    if test -f "experiments/$NAME"; then
      echo "Skipping. $NAME already exists."
    else
      printf "#!/bin/bash\n/usr/bin/python main.py ${model}_${d} mnist $model -c cuda -e 30 -d $d" >> ${model}_${d}.sh
      sbatch --time=08:00:00 --gres=gpu:1 ${model}_${d}.sh
      rm -rf ${model}_${d}.sh
    fi
  done
done

