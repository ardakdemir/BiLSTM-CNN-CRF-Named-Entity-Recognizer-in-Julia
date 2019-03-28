# BiLSTM-CNN-CRF-Named-Entity-Recognizer-in-Julia

Developed by Arda Akdemir, as part of the EU-funded Emerging Welfare Work Package 2 (Information Extraction) in Koc University under the supervision of Deniz Yuret.

We made use of the kNet package to develop our model which is a deep learning package also developed in Koc University.
The model is the reimplementation of the state-of-the-art named entity recognition model [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf).

## Program

main.jl is  the main file of the model. Running the model automatically  trains a model using the data provided in the data folder. 

## Data

The dataset we have used during training and testing are included in the repository train.txt, valid.txt and test.txt.
The dataset is the splitted version of the CoNLL-2003 Named Entity Recognition dataset. 

## Results 

We have obtained comparable results with the reference paper. Our results are included inside repository.
