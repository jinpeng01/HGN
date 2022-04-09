# HGN
Code for Hero-Gang Neural model Accepted in NAACL-2022

## Requirements

- Python 3 (tested on 3.7)
- PyTorch (tested on 1.5 and 1.7)
## Data

We give an example dataset in data/W16_bio


## Training and Evaluation
To start training, run

```
export MODEL=xlnet-large-cased
epoch=20
lr=5e-5
wis=1qq3qq5qq7
data_type=W16_bio
connect_type=dot-att

CUDA_VISIBLE_DEVICES=0 python run_hgn.py \
--train_data_dir=data/$data_type/train_merge.txt \
--dev_data_dir=data/$data_type/dev.txt \
--test_data_dir=data/$data_type/test.txt \
--bert_model=${MODEL} \
--task_name=ner \
--output_dir=./output/xlnet_multi_window_${lr}_win_size_${wis}_epoch_${epoch}_${connect_type} \
--max_seq_length=128 \
--num_train_epochs ${epoch} \
--do_train \
--gpu_id 0 \
--learning_rate ${lr} \
--warmup_proportion=0.1 \s
--train_batch_size=32 \
--use_bilstm \
--use_multiple_window \
--windows_list=${wis} \
--connect_type=${connect_type}
```
In this bash, `Model` is the path to your pre-trained model (such as BERT, XLNET or BioBERT), `windows_list` is the hyperparameter that control the windows.
For example, `1qq3qq5qq7` means that we utilize 4 different windows and their sizes are 1, 3, 5 and 7 respectively. `connect_type` can be mlp-att or dot-att.

