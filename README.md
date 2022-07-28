# 785-project

## Dataset description

- `data/` directory contains raw QB5000 preprocessed and clustered data. A smaller dataset is in `data/2017_01/raw/`,
  and the larger dataset used for this project is in `data/1year/raw/`. Note that these raw data are collected by
  running QB5000 preprocessing pipeline on an AWS machine with 200GB RAM. To further prepare the data for DeepAR and
  baseline LSTM, run

```
cd data/
python preprocess.py
```

## DeepAR

- `src/` contains source code for deepar and baseline LSTM. Implementation of DeepAR and its variants are
  in `src/deepar/`. The source code is adapted from [here](https://github.com/jdb78/pytorch-forecasting)
  and [here](https://github.com/zhykoties/TimeSeries). The other folders
  are mostly used for testing purposes. `src/deepar/main.py` is the main python script for baseline and first two
  extensions. `src/deepar/main_attention.py` is the script for attention DeepAR. The `src/deepar/models/` directory
  contains implementation of the specific models.
- To run baseline DeepAR (all extensions can be run in a similar way):

```
cd src/deepar/
```

To run on colab, do:

```
python main.py \
--data_dir /content/785-project/data/1year \
--data_name 1year \
--colab true \
--ckpt_dir /content/drive/MyDrive/785_project/checkpoints/ \
--plot_dir /content/drive/MyDrive/785_project/plots/
```

To run locally, do:

```
python main.py \
--data_dir ../../data/1year \
--data_name 1year \
--colab false \
--ckpt_dir ./model_checkpoints/v1/ \
--plot_dir ./plots/v1/
```

- To generate evaluation results from trained model, look into `src/deepar/generate_prediction.py`. You can find
  instructions in this script.

## Baseline LSTM

Baseline implementation is in `src/baseline/`.

To train the LSTM model, do

```
cd src/baseline/
python train.py
```

The LSTM models will be saved in the `models` directory and can be loaded if the `override` attribute of
the `ClusterForecaster` class is set to `False`.
The prediction from the LSTM is saved in `src/baseline/prediction`.

The code is adopted from [Ma Lin](https://github.com/malin1993ml/QueryBot5000) based
on [QB5000 paper](http://www.cs.cmu.edu/~malin199/publications/2018.forecasting.sigmod.pdf).

## Evaluation

Evaluation script is in `src/util/validation.py`. This script compares baseline LSTM, DeepAR results with ground truth
and compute evaluation metrics.

To run the evaluation script, do:
```
cd src/util
python validation.py
```
