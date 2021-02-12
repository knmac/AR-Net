GPUS=0  # "0 1 2 3"
BATCH_SIZE=2  # 48
DATA_DIR="./data/ActivityNet"
LOG_DIR="./logs/logs_tsm"
REMOVE_MISSING=1
#check GGGG


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

python main_base.py actnet RGB \
    --arch resnet34 \
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
    --exp_header actnet_res34_t16_epo15_168_lr.001 \
    --rescale_to 168 \
    -j 36 \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --remove_missing $REMOVE_MISSING

python main_base.py actnet RGB \
    --arch resnet18 \
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
    --exp_header actnet_res18_t16_epo10_112_lr.001 \
    --rescale_to 112 \
    -j 36 \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --remove_missing $REMOVE_MISSING

# TODO A-2. joint training for 100 epochs (replace the GGGG with the real datetime shown in your exp dir)
python main_base.py actnet RGB \
    --arch resnet50 \
    --num_segments 16 \
    --lr 0.001 \
    --epochs 100 \
    --batch-size $BATCH_SIZE \
    -j 32 \
    --npb \
    --gpus $GPUS \
    --exp_header jact4_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep100 \
    --ada_reso_skip \
    --policy_backbone mobilenet_v2 \
    --reso_list 224 168 112 84 \
    --backbone_list resnet50 resnet34 resnet18 \
    --skip_list 1 2 4 \
    --accuracy_weight 0.95 \
    --efficency_weight 0.05 \
    --model_paths \
        $LOG_DIR/GGGG_actnet_res50_t16_epo15_224_lr.001/models/ckpt.best.pth.tar \
        $LOG_DIR/GGGG_actnet_res34_t16_epo15_168_lr.001/models/ckpt.best.pth.tar \
        $LOG_DIR/GGGG_actnet_res18_t16_epo15_112_lr.001/models/ckpt.best.pth.tar \
    --exp_decay \
    --init_tau 5 \
    --policy_also_backbone \
    --policy_input_offset 3 \
    --uniform_loss_weight 3.0 \
    --use_gflops_loss \
    --random_seed 1007 \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --remove_missing $REMOVE_MISSING


## TODO B. train for new baseline model (this is also for 100 epochs)
#python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 100 --epochs 120 --batch-size $BATCH_SIZE --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus $GPUS --exp_header actnet_tsn_resnet50_seg16_epo120_sz224_b48_lr.001s100 --rescale_to 224 -j 36 --data_dir $DATA_DIR --log_dir $LOG_DIR
