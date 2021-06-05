# Neural Question Generation with Attention
Master's Thesis: Generating Questions with the Target Answer in Mind
by Sören Etler

## TL;DR
This repository contains the question generation model of my masters thesis. A nodeboo to run the code on google colab is provided `setup_run.ipynb`

## Structure
### Overview

| file          | describtion |
|---------------|-----------|
| main.py       | python script for training a model. This can be done with command line arguments as described below     |
| qg_dataset.py | python script for reading the dataset (e.g. squad, quac)     |
| encoder.py    | Encoder layer of the question generation model |
| decoder.py    | Decoder layer of the question generation model |
| model.py      | Combination of Encoder and Decoder with translation functionality |
| utils.py      | additional functions |

### Command line arguments
| short | long             | type  | default | description |
|-------|------------------|-------|---------|-------------|
| -d    | --dataset        | str   | squad   | the dataset to use for training the model (squad, quac) |
| -t    | --target_length  | int   | 20      | maximum length of the output questions |
| -i    | --input_length   | int   | 80      | maximum length of the input sentences |
| -x    | --vocab_input    | int   | 45000   | size of the vocabulary for the input sentences |
| -y    | --max_vocab_targ | int   | 28000   | size of the vocabulary for the target questions  |
| -e    | --epochs         | int   | 1       | number of epochs to train the model |
| -u    | --units          | int   | 600     | size of the hidden units (if a bidirectional model is used half are forward and half backward units |
| -b    | --batch          | int   | 64      | batch size |
| -l    | --layer          | int   | 1       | number of layers of encoder and decoder |
| -o    | --dropout        | float | 0.3     | dropout in the encoder |
| -p    | --pretrained     | bool  | False   | if pretraine glove wordembeddings are used |
| -r    | --bidirectional  | bool  | True    | if a bidirectional model is used |
| -a    | --answer_units   | int   | 0       | the number of units of the answer encoder |

The size of the answer units determines if a second encoder for the target answer is used.