##!/bin/bash
MODELS=('mos' 'moss' 'sigsoftmax' 'moss' 'plif')

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
#EXPERIMENTS_DIR="experiments/inat"
#DS=('64' '32' '16')
#
#for model in "${MODELS[@]}"
#do
#  for d in "${DS[@]}"
#  do
#    NAME="inat_${model}_k100_d${d}"
#    if test -f "${EXPERIMENTS_DIR}/${NAME}_log"; then
#      echo "Skipping. ${NAME} already exists."
#    else
#      printf "#!/bin/bash\n/usr/bin/python main.py ${NAME} inat $model -c cuda -e 50 -d $d -s 1 -b 128 -t 128 --save_dir ${EXPERIMENTS_DIR}" >> ${NAME}.sh
#      sbatch --time=10:00:00 --mem=20000 --cpus-per-task=4 --gres=gpu:1 ${NAME}.sh
#      rm -rf ${NAME}.sh
#    fi
#  done
#done

