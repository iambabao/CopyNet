# CopyNet

Implement with tensorflow==1.14.  

## Task
I take `Question Generation` task as an example to show how model works.

The data is from [this paper](https://link.springer.com/chapter/10.1007%2F978-3-319-73618-1_56)

You can run:
```bash
python preprocess.py
```
to generate data

## How to use
You can run:
```bash
python run.py \
  --model copynet \
  --do_train \
  --epoch 30 \
  --batch 32 \
  --optimizer Adam \
  --lr 0.001 \
  --dropout 0.2 \
  --pre_train_epochs 5 \
  --early_stop 5 \
  --early_stop_delta 0.001
```

## Acknowledgement
[https://github.com/lspvic/CopyNet](https://github.com/lspvic/CopyNet)
