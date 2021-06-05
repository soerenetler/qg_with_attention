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
| short | long | type | default | description |
| -d    | --dataset | str | squad | display a square of a given number |
| -t    | --target_length | int | 20 | max_length_targ |
| -i    | --input_length | int | 80 | display a square of a given number |
| -x    | --vocab_input | int | 45000 | display a square of a given number |
| -y    | --max_vocab_targ | int | 28000 | display a square of a given number |
| -e    | --epochs | int | 1 | display a square of a given number |
| -u    | --units | int | 600 | display a square of a given number |
| -b    | --batch | int | 64 | display a square of a given number |
| -l    | --layer | int | 1 | display a square of a given number |
| -o    | --dropout | float | default=0.3 | display a square of a given number |
| -p    | --pretrained | bool | False | display a square of a given number |
| -r    | --bidirectional | bool | True | display a square of a given number |
| -a    | --answer_units | int | 0 | display a square of a given number |

The size of the answer units determines if a second encoder for the target answer is used.