#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=MarvinDatasetCSV
DATASET=TomatoWURCSV
dataset_folder=TomatoWUR/data/TomatoWUR/ann_versions/0-paper-2Dto3D/json/
CONFIG=semseg-pt-v3m1-0-base_TOMATOWUR
DATE=`date +%Y%m%d_%H%M`
EXP_NAME=${DATE}_debug
WEIGHT="example_configs/scannet-semseg-pt-v3m1-0-base_model_best_updated.pth"
WEIGHT="None"
RESUME=false
GPU=1


while getopts "p:d:c:n:w:g:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

# EXP_DIR=exp/${DATASET}/${EXP_NAME}
# EXP_DIR=${dataset_folder}/experiments/${DATASET}/trained_models/${EXP_NAME}
EXP_DIR=debug
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/model
# CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
CONFIG_DIR=example_configs/${CONFIG}.py



# echo " =========> CREATE EXP DIR <========="
# echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
# if ${RESUME}
# then
#   CONFIG_DIR=${EXP_DIR}/config.py
#   WEIGHT=$MODEL_DIR/model_last.pth
# else
#   mkdir -p "$MODEL_DIR" "$CODE_DIR"
#   cp -r scripts tools pointcept "$CODE_DIR"
# fi

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r Pointcept/scripts Pointcept/tools Pointcept/pointcept "$CODE_DIR"
fi



echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi