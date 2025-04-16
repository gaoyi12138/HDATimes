model_name=AutoTimes_Gpt2

# training one model with a context length
#torchrun --nnodes 1 --nproc-per-node 8 run.py \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_168_24 \
  --model $model_name \
  --data custom \
  --seq_len 168 \
  --label_len 144 \
  --token_len 24 \
  --test_seq_len 168 \
  --test_label_len 144 \
  --test_pred_len 24 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_activation relu \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --gpu 0 \

# testing the model on all forecast lengths
for test_pred_len in 24 36 48 60
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 168 \
  --label_len 144 \
  --token_len 24 \
  --test_seq_len 168 \
  --test_label_len 144 \
  --test_pred_len $test_pred_len \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_activation relu \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --test_dir long_term_forecast_illness_168_24_AutoTimes_Gpt2_custom_sl168_ll144_tl24_lr0.0001_bt32_wd1e-05_hd1024_hl0_cosTrue_mixTrue_test_0
done