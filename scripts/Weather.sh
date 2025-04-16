model_name=AutoTimes_Gpt2_danet

# training one model with a context length
#torchrun --nnodes 1 --nproc-per-node 4 run.py \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 384 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --lradj type2 \
  --des 'Exp' \
  --mlp_hidden_dim 512 \
  --mlp_hidden_activation relu \
  --mlp_hidden_layers 0 \
  --gpu 0\
  --mix_embeds

# testing the model on all forecast lengths
for test_pred_len in 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 384 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --use_amp \
  --lradj type2 \
  --des 'Exp' \
  --mlp_hidden_dim 512 \
  --mlp_hidden_activation relu \
  --mlp_hidden_layers 0 \
  --mix_embeds \
  --test_dir long_term_forecast_weather_672_96_AutoTimes_Gpt2_danet_custom_sl672_ll576_tl96_lr0.0005_bt384_wd0_hd512_hl0_cosFalse_mixTrue_Exp_0
done