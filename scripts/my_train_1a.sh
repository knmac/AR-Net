GPUS="0 1 2 3"  # "0 1 2 3"
BATCH_SIZE=40  # 48
DATA_DIR="./data/ActivityNet"
LOG_DIR="./logs/logs_tsm"
REMOVE_MISSING=1


# TODO To train to get the improved result for AR-Net(ResNet) (mAP~76.8)
# TODO A. train for new adaptive model
# TODO A-1. prepare each base model (for specific resolution) for 15 epochs
python main_base.py actnet RGB \
    --arch resnet50 \
    --num_segments 16 \
    --gd 20 \
    --lr 0.001 \
    --wd 1e-4 \
    --lr_steps 20 40 \
    --epochs 15 \
    --batch-size $BATCH_SIZE \
    --dropout 0.5 \
    --consensus_type=avg \
    --eval-freq=1 \
    --npb \
    --gpus $GPUS \
    --exp_header actnet_res50_t16_epo15_224_lr.001 \
    --rescale_to 224 \
    -j 36 \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --remove_missing $REMOVE_MISSING
