# FlowQA (and also attention)



## Experiments

The FlowQA model is trained on using the NVIDIA A100 GPU. We train for 30 epochs, have a learning rate of 1 * 10^(-5), with a batch size of 32. We use Python 3.7.1 and PyTorch 1.0.0.


From [FlowQA]https://github.com/momohuang/FlowQA we follow the commands:

Step 1:
perform the following:

> pip install -r requirements.txt

to install all dependent python packages.

Step 2:
download necessary files using:

> ./download.sh

Step 3:
preprocess the data files using:

> python preprocess_QuAC.py

Step 4:
run the training code using:

> python train_QuAC.py

In using attention or not, we distinguish such through these commands:

> python train_QuAC.py --flow_attention=0
> python train_QuAC.py --flow_attention=1

## Training Details
Thesis shows overall statistics of the training/dev/test data split. We would like to evaluate on the test dataset.

## Attention Mechanism
We look to add an attention mechanism in the flow layer. See the flow layer in the FlowQA architecture:
![image](https://github.com/mmKevin-19/seniorthesis-write/assets/72353600/10a29ac1-c7d3-44fa-9d9b-b9e531c89fcb)


## Results

To keep consistency with the epoch number in model training, FlowQA uses 30 epochs for training. On the CoQA test set, the exact matching metric scores 65.6 and is 61.8 for the F1 score metric. For the QuAC test set, we get a F1 score of 59.8. For the FlowQA model with added attention, the scores for exact matching and F1 score on CoQA test set is 65.5 and 63.3 respectively, and the F1 score is 59.6 for the QuAC test set. Marginal improvement with attention implementation. 
![image](https://github.com/mmKevin-19/seniorthesis-write/assets/72353600/5f6e3ff4-36d1-4d1a-ba5d-3a416cb2c577)

