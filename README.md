# Prediction of Synthetic Lethal Interactions in Human Cancers using Multi-view Graph Auto-Encoder

#### Requirements

1. Python == 3.7
2. Tensorflow == 1.15.0
3. numpy == 1.18.4
4. scipy == 1.4.1
5. pandas == 1.0.3

#### Repository Structure

- SLMGAE/BC_data/: SynLethDB Breast-Cancer datasets.
- SLMGAE/data/: SynLethDB datasets.
- SLMGAE/code/: Our SLMGAE model.

#### How to run our code

- train.py  # Train and evaluate the SLMGAE model base on SynLeth data
- train_BC.py # Train and evaluate the SLMGAE model base on SynLeth-data model
- case_study.py # Train our SLMGAE using all the SL pairs in SynLethDB and predict novel SLs from the unknown pairs.
