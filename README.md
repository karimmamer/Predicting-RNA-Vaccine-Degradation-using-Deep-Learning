# RNA Vaccine Degradation Prediction using Deep Neural Networks

13th place solution for OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction challnge organized by Stanford and Kaggle.

## Getting Started

A summarized description of the approach can be found [here](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189585).

## Prerequisites

* Python3
* Arnie
* bpRNA
* Pytorch
* Scikit-learn
* tqdm

Arnie and bpRNA folders should be in the same directory as the source code.

## Running

### Extra Dataset Generation

```
python3 generate_extra_dataset.py --data_path ... --package ... --temp ... --n_threads ...
```

This step generates a number of secondary structures per sequence (default is 5) using a package from Arnie to be used for augmentation. You need to choose a package, a temperature at which bpp matrices are generated and the number of threads to use in parallel.

### Single Model Training

```
python3 main.py --data_path ... --arch ... --package ... --temp ...
```

This step trains a single model with 5-fold validation and creates a submission file. You need to choose an architecture (--arch) from cnn, cnn_lstm or cnn_lstm_transformer.

### Ensemble

To create an ensemble, you need to generate structures from different packages using differnet temperatures, train different architecures with them and finally average the resulted submissions.

