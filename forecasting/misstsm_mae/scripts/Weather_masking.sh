DATASET="weather"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

BASE_PATH="./data/synthetic_datasets/weather/"
ROOT_PATHS=$1
DEVICE=$2
TRIAL=$3
MASKINGTYPE=$4
PRED_LEN_LIST=$5

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_weather.csv"

GT_SOURCE_FILE="weather.csv"
GT_ROOT_PATH="./data/forecasting_datasets/weather/"

OUTPUT_PATH="./outputs/${MASKINGTYPE}/weather_v${TRIAL}/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    
    root_path="${BASE_PATH}${id}"
    # PRETRAIN
    python -u executor.py \
        --task_name pretrain \
        --device $DEVICE \
        --root_path $root_path \
        --run_name "v${TRIAL}_${MASKINGTYPE}_new_pretrain_${DATASET}_${id}" \
        --source_filename $SOURCE_FILE \
        --dataset $DATASET \
        --max_epochs $PRETRAIN_EPOCHS \
        --mask_ratio 0.50 \
        --encoder_depth 2 \
        --encoder_num_heads 8 \
        --lr 0.001 \
        --batch_size 16 \
        --encoder_embed_dim 64 \
        --decoder_embed_dim 32 \
        --decoder_num_heads 4 \
        --decoder_depth 2 \
        --project_name weather_masking \
        --trial $TRIAL \
        --dropout 0.001
    
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path \
            --gt_root_path $GT_ROOT_PATH \
            --gt_source_filename $GT_SOURCE_FILE \
            --run_name "v${TRIAL}_${MASKINGTYPE}_new_finetune_${DATASET}_PRED_${pred_len}_${id}" \
            --pretrain_run_name "v${TRIAL}_${MASKINGTYPE}_new_pretrain_${DATASET}_${id}" \
            --freeze_encoder "False" \
            --max_epochs $FINETUNE_EPOCHS \
            --dataset $DATASET \
            --pred_len $pred_len \
            --source_filename $SOURCE_FILE \
            --pretrain_ckpt_name ckpt_best.pth \
            --encoder_depth 2 \
            --encoder_num_heads 8 \
            --encoder_embed_dim 64 \
            --lr 0.0001 \
            --dropout 0.001 \
            --fc_dropout 0.0182 \
            --batch_size 16 \
            --accum_iter 1 \
            --project_name weather_masking \
            --output_path $OUTPUT_PATH \
            --trial $TRIAL
    done
done