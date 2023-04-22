##!/bin/bash
MODELS=('softmax' 'mos' 'sigsoftmax' 'moss' 'plif')

# Run MNIST
EXPERIMENTS_DIR="experiments/mnist"
DS=('128' '9' '8' '7' '6' '5' '4' '3' '2' '1')

for model in "${MODELS[@]}"
do
  for d in "${DS[@]}"
  do
    NAME="mnist_${model}_${d}"
    if test -f "${EXPERIMENTS_DIR}/${NAME}_log"; then
      echo "Skipping. ${NAME} already exists."
    else
      printf "#!/bin/bash\n/usr/bin/python main.py ${NAME} mnist $model -c cuda -e 40 -d $d -s 10 --save_dir ${EXPERIMENTS_DIR}" >> ${NAME}.sh
      sbatch --time=7:00:00 --gres=gpu:1 ${NAME}.sh
      rm -rf ${NAME}.sh
    fi
  done
done

# Run INaturalist
EXPERIMENTS_DIR="experiments/inat"
DS=('32' '16' '8' '4' '2' '1')

for model in "${MODELS[@]}"
do
  for d in "${DS[@]}"
  do
    NAME="inat_${model}_k100_d${d}"
    if test -f "${EXPERIMENTS_DIR}/${NAME}_log"; then
      echo "Skipping. ${NAME} already exists."
    else
      printf "#!/bin/bash\n/usr/bin/python main.py ${NAME} inat $model -c cuda -e 200 -d $d -s 10 -b 128 -t 10000 --save_dir ${EXPERIMENTS_DIR}" >> ${NAME}.sh
      sbatch --time=7:00:00 --gres=gpu:1 ${NAME}.sh
      rm -rf ${NAME}.sh
    fi
  done
done

# Impact of the number of mixtures
EXPERIMENTS_DIR="experiments/mixtures"
MS=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100' '110' '120' '130' '140' '150' '160' '170' '180' '190' '200')

for m in "${MS[@]}"
do
  NAME="inat_mos_k100_d4_m${m}"
  if test -f "${EXPERIMENTS_DIR}/${NAME}_log"; then
    echo "Skipping. ${NAME} already exists."
  else
    printf "#!/bin/bash\n/usr/bin/python main.py ${NAME} inat mos -c cuda -e 200 -d 4 -s 10 -b 128 -t 10000 -m $m --save_dir ${EXPERIMENTS_DIR}" >> ${NAME}.sh
    sbatch --time=7:00:00 --gres=gpu:1 ${NAME}.sh
    rm -rf ${NAME}.sh
  fi
done
