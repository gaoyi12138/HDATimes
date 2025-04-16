model_name=AutoTimes_Gpt2_danet

# training one model with a context length
#torchrun --nnodes 1 --nproc-per-node 8 run.py \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path wind_power.csv \
  --model_id wind_power_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.000001 \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_activation swiglu \
  --mlp_hidden_layers 0 \
  --train_epochs 20 \
  --gpu 0 \
  --mix_embeds \
  --tmax 10 \
  --cosine

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/wind/ \
  --data_path wind_power.csv \
  --model_id wind_power_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 8 \
  --learning_rate 0.000001 \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_activation swiglu \
  --train_epochs 10 \
  --gpu 0 \
  --mix_embeds \
  --tmax 10 \
  --cosine \
  --test_dir long_term_forecast_wind_power_672_96_AutoTimes_Gpt2_danet_custom_sl672_ll576_tl96_lr1e-06_bt256_wd0_hd1024_hl0_cosTrue_mixTrue_test_0
done