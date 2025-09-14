export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="./forecasting_datasets/ETT/"
root_path_name="./synthetic_datasets/ETTm2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm2_imputed_SAITS.csv"

OUTPUT_PATH="./outputs/SAITS/${MASKINGTYPE}/ETTm2_v${TRIAL}/"
CHECKPOINT="./AutoFormer_ckpts/SAITS/"

seq_len=336

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path ETTm2.csv \
          --model_id "ETTm2_336_${pred_len}" \
          --model Autoformer \
          --data ETTm2 \
          --features M \
          --seq_len $seq_len \
          --label_len 168 \
          --pred_len $pred_len \
          --e_layers 2 \
          --d_layers 1 \
          --factor 1 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --freq 't' \
          --itr 1 \
          --gpu $DEVICES \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done