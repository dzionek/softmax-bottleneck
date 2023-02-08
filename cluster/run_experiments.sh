##!/bin/bash
MODELS=('softmax' 'sigsoftmax' 'plif')
DS=('128' '5' '3')

for model in "${MODELS[@]}"
do
  for d in "${DS[@]}"
  do
    printf "#!/bin/bash\n/usr/bin/python ${model}_${d} mnist $model -c cuda -e 30 -d $d" >> ${model}_${d}.sh
    sbatch --time=08:00:00 --gres=gpu:1 ${model}_${d}.sh
    rm -rf ${model}_${d}.sh
  done
done

