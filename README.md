#HDATimes

The repo is the official implementation for the paper: [A Wind Power Forecasting Method Based on Large Language Models]. 

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. Put the datasets [[Google Drive]](https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view?usp=sharing)
[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/849427d3926f4fabbee7/) under the folder ```./dataset/```.

2. Download the large language models from [Hugging Face](https://huggingface.co/) and specify the model path using the `llm_ckp_dir` parameter in scripts.
   * [GPT2](https://huggingface.co/openai-community/gpt2)
   * [OPT Family](https://huggingface.co/facebook/opt-125m)
   * [LLaMA-7B](https://huggingface.co/meta-llama/Llama-2-7b)

3. Train and evaluate the model. We provide all the above tasks under the folder ```./scripts/```.

```
# the default large language model is LLaMA-7B

# long-term forecasting
bash ./scripts/time_series_forecasting/long_term/AutoTimes_ETTh1.sh



## Citation

If you find this repo helpful, please cite our paper. 
