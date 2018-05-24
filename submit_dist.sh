export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=64
export USE_PRUNING=use_pruning
export USE_RESIDUE_ACC=use_residue_acc
export USE_WARMUP=use_warmup
export USE_SYNC=no_use_sync
export MODEL_NAME=resnet
#export MODEL_NAME=mobilenetv2 #resnet
export RESNET_DEPTH=44
export LRSCALE=lr_bb_fix
# a Vanilla SGD, useless because py in models decides mom
export MOMENTUM=0.9
export WEIGHTDECAY=0.0001
export USE_NES=use_nesterov
export DATASET=cifar10

mpirun -np 2 python3 ./main_dist_dgc.py \
  --gpus 3,4 \
  --dataset ${DATASET} \
  --resnet_depth=${RESNET_DEPTH} \
  --model ${MODEL_NAME} \
  --epochs 41 \
  --b ${BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --weight-decay=${WEIGHTDECAY} \
  --${USE_RESIDUE_ACC} \
  --${USE_NES} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${LRSCALE} \
  --save dist_${DATASET}_${MODEL_NAME}${RESNET_DEPTH}_${BATCH_SIZE}_${USE_PRUNING}_${USE_RESIDUE_ACC}_${USE_WARMUP}_${USE_SYNC}_${USE_NES}_allgather


