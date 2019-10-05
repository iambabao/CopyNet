# copynet

A tensorflow implementation of CopyNet

## Implementation
Implement with tensorflow==1.14.  
Some methods will be **DEPRECATED**, but stil work in this version. 

## Task
I take `Question Generation` task as an example to show how model works.

The dataset is [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).  
The input is the concatenation of sentences containing answer and answer itself.  
The output is question. More detail can be found in `preprocess.py` and `data_reader.py`.

## How to use
run `train.py` with arguments:
```bash
python train.py -m copynet --batch 32 --epoch 30 --optimizer custom
```

run `test.py` with arguments
```bash
python test.py -m copynet --batch 32 --optimizer custom
```
More detail can be found in `train.py` and `test.py`.

Traditional `seq2seq` model with attention mechanism is also available.

## Acknowledgement
Most of code in `copynet_wrapper.py` are copied from [https://github.com/lspvic/CopyNet](https://github.com/lspvic/CopyNet)
