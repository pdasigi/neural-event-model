# Neural Event Model

Code for training and testing Neural Event Model (NEM). We represent events as composition of the main verb in the sentence and its semantic role filler arguments. This is a supervised model, and composition, word and event representations are learned based on the end objective.

## Requirements

This code depends on Keras 2.0.3, and is written in Python 3.5.

## Data Format

Train and test data is expected in JSON format with the following fields
```
[
  {
    "sentence": "string",
    "event_structure": {
      "V": "string",
      "A0": "string",
      ...
    }
    "label": 0
  }
]
```
The dataset is a list of dicts, with each dict containing `sentence`, `event_structure` and a `label`. `event_structire` is a dict containing the verb and semantic role fillers. We use Propbank style SRL tags. Label is either 0 or 1.

## Training

`python nem.py --train_file train.json --test_file test.json`

or alternatively

`python nem.py --train_file train.json --test_file test.json --embedding_file embedding.gz`

if you want to use pretrained embeddings. Run `python nem.py -h` for more options.

## Paper

This is a reimplementation (with minor modifications) of the model described in the following paper, with labels indicating newswire anomalies:

[*Modeling Newswire Events using Neural Networks for Anomaly Detection*](http://www.anthology.aclweb.org/C/C14/C14-1134.pdf)

