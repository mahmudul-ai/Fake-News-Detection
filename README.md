# Fake News Detection Using Linguistic and Behavioral Biometrics

The dependencies are available at `requirements.txt`. Run the following command to install them:

    pip install -r requirements.txt


`LIAR2` folder contains the LIAR2 dataset.

`LIAR2+Biometric` folder contains the LIAR2 dataset with the calculated biometric features.

Use `biometric_feature_extraction.py` to extract the biometric features of the LIAR2 dataset.

Use `model_bio.py` to implement the FDHN model architecture.

Use `data_loader_biometric.py` to load the data. Modify `lines 142-150` to fix which numerical metadata to consider in training.

Use `train_bio.py` to train the model.

Modify `config-biometric.json` to adjust training settings. The value of `input_dim_metadata` is the length of `num_cols` in `lines 142-150` of `data_loader_biometric.py`.

Use `test_bio.py` to test the performance of the model.
