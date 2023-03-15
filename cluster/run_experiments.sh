##!/bin/bash
MODELS=('softmax' 'mos' 'sigsoftmax' 'moss' 'plif')
DS=('128' '9' '8' '7' '6' '5' '4' '3' '2' '1')

# Run MNIST
EXPERIMENTS_MNIST_DIR="experiments/mnist"
for model in "${MODELS[@]}"
do
  for d in "${DS[@]}"
  do
    NAME="mnist_${model}_${d}"
    if test -f "${EXPERIMENTS_MNIST_DIR}/${NAME}_log"; then
      echo "Skipping. ${NAME} already exists."
    else
      printf "#!/bin/bash\n/usr/bin/python main.py ${NAME} mnist $model -c cuda -e 40 -d $d -s 10 --save_dir ${EXPERIMENTS_MNIST_DIR}" >> ${NAME}.sh
      sbatch --time=7:00:00 --gres=gpu:1 ${NAME}.sh
      rm -rf ${NAME}.sh
    fi
  done
done

