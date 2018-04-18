export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=128
export MINI_BATCH_SIZE=128
export USE_PRUNING=no_use_pruning
export USE_RESIDUE_ACC=use_residue_acc
export USE_WARMUP=use_warmup
export USE_SYNC=no_use_sync
export MODEL_NAME=resnet
export RESNET_DEPTH=44 #44
export LRSCALE=lr_bb_fix
# a Vanilla SGD, useless because py in models decides mom
export MOMENTUM=0.9
export WEIGHTDECAY=1e-4

python3 main_dgc.py \
  --gpus 2,4 \
  --dataset cifar10 \
  --resnet_depth=${RESNET_DEPTH} \
  --model ${MODEL_NAME} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --mini-batch-size ${MINI_BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --weight-decay=${WEIGHTDECAY} \
  --${USE_RESIDUE_ACC} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${LRSCALE} \
  --save debug

#--use_pruning ${USE_PRUNING}
#--use_residue_acc ${USE_RESIDUE_ACC}
#--resume ./TrainingResults/momentum_cifar10_resnet_128_64_use_pruning_use_residue_acc_use_warmup_no_use_sync/checkpoint.pth.tar
