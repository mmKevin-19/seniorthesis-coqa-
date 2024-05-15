# RoBERTa: A Robustly Optimized BERT Pretraining Approach

https://arxiv.org/abs/1907.11692
We use same preprocess scripts as in the upper directory and in FlowQA.

## Introduction

RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. RoBERTa excels on downstream tasks such as question answering.

##### Load RoBERTa (for PyTorch 1.0 or custom models):
```python
# Download roberta.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz


### Preprocess

Data should be preprocessed following the [language modeling format](/examples/language_model), i.e. each document should be separated by an empty line (only useful with `--sample-break-mode complete_doc`). Lines will be concatenated as a 1D text stream during training.

First download the dataset:
In this case, we load the CoQA and QuAC dataset.

Next encode it with the GPT-2 BPE:
```bash
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```

##  Historical Embeddings
We also test RoBERTa with historical embeddings. Embeddings are incorporated in the following way and can be seen in preprocessRob.py.
![image](https://github.com/mmKevin-19/seniorthesis-write/assets/72353600/99cc7b91-c7b2-42cb-ab93-6fc47e1424ca)



## Experiments

The RoBERTa model is trained on using the NVIDIA A100 GPU. We train for 30 epochs, have a learning rate of 1 * 10^(-5), with a batch size of 32. The model was run with PyTorch 0.4.1 and spaCy 2.0.16. 


## Result 
On the CoQA test set, the exact matching and F1 scores are 60.8 and 55.8 for RoBERTa respectively. For the QuAC test set, we get a F1 score of 50.8. When we add historical  embeddings, we get 62.3, 59.6 as our exact matching and F1 scores. For QuAC, our F1 score is 52.3. We see noticeable improvement in historical embedding usage.

![image](https://github.com/mmKevin-19/seniorthesis-write/assets/72353600/971ebc9f-ad3e-4cf7-9956-6738225673d6)
