

# SDNet
SDNet is an attention based deep neural network that emphasizes contextualization through recurrent neural networks. This model was originally only evaluated on CoQA, but we look to evaluate on the QuAC dataset as well. 

We look to ensure compatibility of the datasets. We would like to make sure that the inputs on SDNet that work for CoQA are compatible with QuAC as well. The fundamental difference between these two datasets is CoQA does not have the “follow up” property, where the student has the choice to continue asking questions in a conversation consisting of multiple questions and answers. Additionally, QuAC does not have to deal with the “rational” section that CoQA does. Finally, we also need to consider “Yes” and “No” as categorical answers for QuAC. Whereas CoQA has “Yes” and “No” as actual answers, QuAC uses “Yes” and “No” as categories for if the answer to a question in QuAC is generally “Yes” or “No”, followed by a longer answer. 


# Architecture

See architecture below. Contextualized based neural network with recurrenct neural networks as the foundation.
![image](https://github.com/mmKevin-19/seniorthesis-write/assets/72353600/3929b37a-041b-47da-85b8-d4e217c2f6ac)


## Difference between CoQA and QuAC
Since the format of dataset is different between CoQA and QuAC, we need to convert the format first to let SDNet work on QuAC. Unlike QuAC, the answers in CoQA are context-free. However, SDNet still need to generate an answer span from text, so we can take this as output on QuAC. Moreover, questions in CoQA don’t have properties such as “follow up” and “yes/no”, but all questions are possible to be related to previous questions and answers. Original SDNet model prepends 2 rounds of previous questions and answers to obtain the best F1 score, so we will keep this behavior and ignore “follow up” property. Also, SDNet essentially needs to compute the probability of answer being affirmation “yes” or “no”, and “No answer” (“unknown” in CoQA), which we can use directly on QuAC.

## Experiments

The SDNet model is trained on using the NVIDIA A100 GPU. We train for 30 epochs, have a learning rate (??) of 1 * 10^(-5), with a batch size of 32. The model was run with PyTorch 0.4.1 and spaCy 2.0.16. 




## Usage
See preprocesscoqa.py for conversion of QuAC data to SDNet input format.


## Result 
On the CoQA test set, the exact matching metric scores a 63.8 and is 59.5 for the F1 score metric. For the QuAC test set, we get a F1 score of 38.2. 

![image](https://github.com/mmKevin-19/seniorthesis-write/assets/72353600/1b73b191-2603-47ec-8562-c898eb9cb2d0)



