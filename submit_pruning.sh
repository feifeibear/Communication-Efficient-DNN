export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=256
export MINI_BATCH_SIZE=64
export USE_PRUNING=use_pruning #True
export USE_RESIDUE_ACC=use_residue_acc  #True
export USE_WARMUP=use_warmup
export USE_SYNC=no_use_sync
export PRUNING_PERC=0.01
export MODEL_NAME=resnet
export MOMENTUM=1.0

python3 main_gbn.py \
  --gpus 1 \
  --dataset cifar10 \
  --model ${MODEL_NAME} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --mini-batch-size ${MINI_BATCH_SIZE} \
  --pruning_perc ${PRUNING_PERC} \
  --momentum=${MOMENTUM} \
  --${USE_RESIDUE_ACC} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --no-lr_bb_fix \
  --save no_momentum_cifar10_${MODEL_NAME}_${BATCH_SIZE}_${MINI_BATCH_SIZE}_${USE_PRUNING}_${USE_RESIDUE_ACC}_${USE_WARMUP}_${USE_SYNC}

#--use_pruning ${USE_PRUNING} \
#--use_residue_acc ${USE_RESIDUE_ACC} \
#--resume ./TrainingResults/momentum_cifar10_resnet_128_64_use_pruning_use_residue_acc_use_warmup_no_use_sync/checkpoint.pth.tar \
