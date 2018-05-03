export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=128
export MINI_BATCH_SIZE=32
export USE_PRUNING=no_use_pruning #True
export USE_RESIDUE_ACC=no_use_residue_acc  #True
export USE_WARMUP=no_use_warmup
export USE_SYNC=no_use_sync
export USE_DELAYED_SGD=no_use_delayed_sgd
#export MODEL_NAME=lenet #mobilenetv2 #resnet
#export MODEL_NAME=mobilenetv2 #resnet
export MODEL_NAME=resnet #resnet
export RESNET_DEPTH=44
export LRSCALE=lr_bb_fix
# a Vanilla SGD
export MOMENTUM=0.9
export WEIGHTDECAY=0.0001

python3 main_asyn_dgc.py \
  --gpus 3 \
  --dataset cifar10 \
  --model ${MODEL_NAME} \
  --resnet_depth=${RESNET_DEPTH} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --mini-batch-size ${MINI_BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --weight-decay=${WEIGHTDECAY} \
  --${USE_RESIDUE_ACC} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${USE_DELAYED_SGD} \
  --lr_bb_fix \
  --save asyn_cifar10_${MODEL_NAME}_${RESNET_DEPTH}_${BATCH_SIZE}_${MINI_BATCH_SIZE}_${USE_DELAYED_SGD}
  #--save base_cifar10_${MODEL_NAME}_${RESNET_DEPTH}_${BATCH_SIZE}_${MINI_BATCH_SIZE}_myMSGD

#--use_pruning ${USE_PRUNING} \
#--use_residue_acc ${USE_RESIDUE_ACC} \
#--resume ./TrainingResults/momentum_cifar10_resnet_128_64_use_pruning_use_residue_acc_use_warmup_no_use_sync/checkpoint.pth.tar \
