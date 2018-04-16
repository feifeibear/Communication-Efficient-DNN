export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=128
export MINI_BATCH_SIZE=32
export USE_PRUNING=use_pruning
export USE_RESIDUE_ACC=use_residue_acc
export USE_WARMUP=use_warmup
export USE_SYNC=no_use_sync
export MODEL_NAME=resnet
export RESNET_DEPTH=44 #44
export LRSCALE=lr_bb_fix
# a Vanilla SGD, useless because py in models decides mom
export MOMENTUM=0.0

python3 main_dgc.py \
  --gpus 0,1 \
  --dataset cifar10 \
  --resnet_depth=${RESNET_DEPTH} \
  --model ${MODEL_NAME} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --mini-batch-size ${MINI_BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --${USE_RESIDUE_ACC} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${LRSCALE} \
  --save dgc_cifar10_${MODEL_NAME}${RESNET_DEPTH}_${BATCH_SIZE}_${MINI_BATCH_SIZE}_${USE_PRUNING}_${USE_RESIDUE_ACC}_${USE_WARMUP}_${USE_SYNC}_fixedUVbug_selfwd

#--use_pruning ${USE_PRUNING}
#--use_residue_acc ${USE_RESIDUE_ACC}
#--resume ./TrainingResults/momentum_cifar10_resnet_128_64_use_pruning_use_residue_acc_use_warmup_no_use_sync/checkpoint.pth.tar
