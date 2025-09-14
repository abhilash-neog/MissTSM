export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # specify visible GPUs

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="./forecasting_datasets/weather/"
root_path_name="./synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed_SAITS.csv"

OUTPUT_PATH="./outputs/SAITS/${MASKINGTYPE}/weather_v${TRIAL}/"
CHECKPOINT="./DLinear_ckpts/SAITS/"

seq_len=336

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path "$root_path" \
          --data_path "$data_path_name" \
          --gt_root_path "$GT_ROOT_PATH" \
          --gt_data_path weather.csv \
          --model_id "Weather_${seq_len}_${pred_len}" \
          --model DLinear \
          --data weather \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --des 'Exp' \
          --itr 1 \
          --batch_size 16 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done