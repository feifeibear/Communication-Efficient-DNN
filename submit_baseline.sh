export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=128
export MINI_BATCH_SIZE=128
export USE_PRUNING=no_use_pruning #True
export RESNET_DEPTH=44
export USE_RESIDUE_ACC=use_residue_acc  #True
export USE_WARMUP=no_use_warmup
export USE_SYNC=no_use_sync
export PRUNING_PERC=0.01
export MODEL_NAME=resnet
# a Vanilla SGD
export MOMENTUM=0.9

python3 main_dgc.py \
  --gpus 3,4 \
  --dataset cifar10 \
  --model ${MODEL_NAME} \
  --resnet_depth=${RESNET_DEPTH} \
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
  --save base_cifar10_${MODEL_NAME}_${RESNET_DEPTH}_${BATCH_SIZE}_${MINI_BATCH_SIZE}_mom${MOMENTUM}_imp_mom_weight_decay_SGD_myself

#--use_pruning ${USE_PRUNING} \
#--use_residue_acc ${USE_RESIDUE_ACC} \
#--resume ./TrainingResults/momentum_cifar10_resnet_128_64_use_pruning_use_residue_acc_use_warmup_no_use_sync/checkpoint.pth.tar \
