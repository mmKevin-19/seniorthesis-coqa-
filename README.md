# Thesis Code

# Conversational Question Answering Systems
There are two datasets used here: CoQA (Conversational Question Answering Dataset) and QuAC (Question Answering in Context). 

We look to evaluate on these datasets with the following three models: RoBERTa, SDNet, and FlowQA. See the respective model directories for further elaboration and code.

# CoQA dataset

Use the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) library for all seq2seq experiments, and clone repository.

```bash
  git clone --recurse-submodules git@github.com:stanfordnlp/coqa-baselines.git
```



## Requirements
```
torch>=0.4.0
torchtext==0.2.1
gensim
pycorenlp
```

## Download
Download the dataset:
```bash
  mkdir data
  wget -P data https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
  wget -P data https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
```

Download pre-trained word vectors:
```bash
  mkdir wordvecs
  wget -P wordvecs http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
  unzip -d wordvecs wordvecs/glove.42B.300d.zip
  wget -P wordvecs http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
  unzip -d wordvecs wordvecs/glove.840B.300d.zip
```

## Start a CoreNLP server

```bash
  mkdir lib
  wget -P lib http://central.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.9.1/stanford-corenlp-3.9.1.jar
  java -mx4g -cp lib/stanford-corenlp-3.9.1.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Conversational models
### Preprocessing
Generate the input files for seq2seq models --- needs to start a CoreNLP server (`n_history` can be changed to {0, 1, 2, ..} or -1):
```bash
  python scripts/gen_seq2seq_data.py --data_file data/coqa-train-v1.0.json --n_history 2 --lower --output_file data/seq2seq-train-h2
  python scripts/gen_seq2seq_data.py --data_file data/coqa-dev-v1.0.json --n_history 2 --lower --output_file data/seq2seq-dev-h2
```

Preprocess the data and embeddings:
```bash
  python seq2seq/preprocess.py -train_src data/seq2seq-train-h2-src.txt -train_tgt data/seq2seq-train-h2-tgt.txt -valid_src data/seq2seq-dev-h2-src.txt -valid_tgt data/seq2seq-dev-h2-tgt.txt -save_data data/seq2seq-h2 -lower -dynamic_dict -src_seq_length 10000
  PYTHONPATH=seq2seq python seq2seq/tools/embeddings_to_torch.py -emb_file_enc wordvecs/glove.42B.300d.txt -emb_file_dec wordvecs/glove.42B.300d.txt -dict_file data/seq2seq-h2.vocab.pt -output_file data/seq2seq-h2.embed
```

### Training
Run a seq2seq (with attention) model:
```bash
   python seq2seq/train.py -data data/seq2seq-h2 -save_model seq2seq_models/seq2seq -word_vec_size 300 -pre_word_vecs_enc data/seq2seq-h2.embed.enc.pt -pre_word_vecs_dec data/seq2seq-h2.embed.dec.pt -epochs 50 -gpuid 0 -seed 123

### Testing
```bash
  python seq2seq/translate.py -model seq2seq_models/seq2seq_copy_acc_65.49_ppl_4.71_e15.pt -src data/seq2seq-dev-h2-src.txt -output seq2seq_models/pred.txt -replace_unk -verbose -gpu 0
  python scripts/gen_seq2seq_output.py --data_file data/coqa-dev-v1.0.json --pred_file seq2seq_models/pred.txt --output_file seq2seq_models/seq2seq_copy.prediction.json
```


## Reading comprehension models
### Preprocessing
Generate the input files for the reading comprehension (extractive question answering) model -- needs to start a CoreNLP server:
```bash
  python scripts/gen_drqa_data.py --data_file data/coqa-train-v1.0.json --output_file coqa.train.json
  python scripts/gen_drqa_data.py --data_file data/coqa-dev-v1.0.json --output_file coqa.dev.json
```

### Training
`n_history` can be changed to {0, 1, 2, ..} or -1.
```bash
  python rc/main.py --trainset data/coqa.train.json --devset data/coqa.dev.json --n_history 2 --dir rc_models --embed_file wordvecs/glove.840B.300d.txt
```


### Testing
```bash
  python rc/main.py --testset data/coqa.dev.json --n_history 2 --pretrained rc_models
```

## The pipeline model
### Preprocessing
```bash
  python scripts/gen_pipeline_data.py --data_file data/coqa-train-v1.0.json --output_file1 data/coqa.train.pipeline.json --output_file2 data/seq2seq-train-pipeline
  python scripts/gen_pipeline_data.py --data_file data/coqa-dev-v1.0.json --output_file1 data/coqa.dev.pipeline.json --output_file2 data/seq2seq-dev-pipeline
  python seq2seq/preprocess.py -train_src data/seq2seq-train-pipeline-src.txt -train_tgt data/seq2seq-train-pipeline-tgt.txt -valid_src data/seq2seq-dev-pipeline-src.txt -valid_tgt data/seq2seq-dev-pipeline-tgt.txt -save_data data/seq2seq-pipeline -lower -dynamic_dict -src_seq_length 10000
  PYTHONPATH=seq2seq python seq2seq/tools/embeddings_to_torch.py -emb_file_enc wordvecs/glove.42B.300d.txt -emb_file_dec wordvecs/glove.42B.300d.txt -dict_file data/seq2seq-pipeline.vocab.pt -output_file data/seq2seq-pipeline.embed
```

### Training
`n_history` can be changed to {0, 1, 2, ..} or -1.
```bash
  python rc/main.py --trainset data/coqa.train.pipeline.json --devset data/coqa.dev.pipeline.json --n_history 2 --dir pipeline_models --embed_file wordvecs/glove.840B.300d.txt --predict_raw_text n
  python seq2seq/train.py -data data/seq2seq-pipeline -save_model pipeline_models/seq2seq_copy -copy_attn -reuse_copy_attn -word_vec_size 300 -pre_word_vecs_enc data/seq2seq-pipeline.embed.enc.pt -pre_word_vecs_dec data/seq2seq-pipeline.embed.dec.pt -epochs 50 -gpuid 0 -seed 123
```

### Testing
```bash
  python rc/main.py --testset data/coqa.dev.pipeline.json --n_history 2 --pretrained pipeline_models
  python scripts/gen_pipeline_for_seq2seq.py --data_file data/coqa.dev.pipeline.json --output_file pipeline_models/pipeline-seq2seq-src.txt --pred_file pipeline_models/predictions.json
  python seq2seq/translate.py -model pipeline_models/seq2seq_copy_acc_85.00_ppl_2.18_e16.pt -src pipeline_models/pipeline-seq2seq-src.txt -output pipeline_models/pred.txt -replace_unk -verbose -gpu 0
  python scripts/gen_seq2seq_output.py --data_file data/coqa-dev-v1.0.json --pred_file pipeline_models/pred.txt --output_file pipeline_models/pipeline.prediction.json
```

## Results

All the results are based on `n_history = 2`:

| Model  | EM | F1 |
| ------------- | ------------- | ------------- |
| seq2seq | 20.9 | 17.7 |
| DrQA | 55.6 | 46.2 |
| RoBERTa | 60.8 | 55.8 |
| RoBERTa + history embeddings | 62.3 | 59.6 |
| SDNet | 63.8 | 59.5 |
| FlowQA | 65.6 | 61.8 |
| FlowQA + attention on flow layer | 65.5 | 63.3 |

# QuAC dataset

We process the QuAC dataset through the quacmetric.py, quacprocess.py, and quacrun.py files. We do not test for DrQA model in the QuAC dataset and also do not test for exact matching. QuAC test set results are as shown:

| Model  | F1 |
| ------------- | ------------- |
| seq2seq | 19.7 |
| RoBERTa | 50.8 |
| RoBERTa + history embeddings | 52.3|
| SDNet | 38.2 |
| FlowQA | 59.8 |
| FlowQA + attention on flow layer | 59.3 |


## Citation

```
    @article{reddy2019coqa,
      title={{CoQA}: A Conversational Question Answering Challenge},
      author={Reddy, Siva and Chen, Danqi and Manning, Christopher D},
      journal={Transactions of the Association of Computational Linguistics (TACL)},
      year={2019}
    }

@misc{choi2018quac,
      title={QuAC : Question Answering in Context}, 
      author={Eunsol Choi and He He and Mohit Iyyer and Mark Yatskar and Wen-tau Yih and Yejin Choi and Percy Liang and Luke Zettlemoyer},
      year={2018},
      eprint={1808.07036},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
MIT
