# PeaCap: Introducing Image Patching for Retrieval-Augmented Image Captioning

## Setup
Install the required packages using conda with the provided [environment.yaml](environment.yaml) file.

## Training
Train PeaCap on the COCO training dataset, using the [scripts/train.sh](scripts/train.sh) script.

## Training & Validation
Train PeaCap on the COCO training dataset and evaluate on the COCO Karpathy validation test set, using [scripts/train_eval.sh](scripts/train_eval.sh) script.

## Evaluation
Evaluate the trained PeaCap on the COCO Karpathy test set, using [scripts/eval.sh](scripts/eval.sh) script.

## Acknowledgements
This repo is built on [EVCap](github.com/Jiaxuan-Li/EVCap). We thank the authors for their great effort and inspiration.
