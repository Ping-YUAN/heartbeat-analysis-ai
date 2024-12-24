# Project Name

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

To understand the file structure check [Project Organization](./organization.md).

The dataset comes from kaggle [here](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data).

To run this project, you need to download those 4 files, and put them under `data/raw` folder with the original name.

Recommand using conda to management the enviorment,

Recoomand intercepter version is: 3.10.14
`conda install --yes --file requirements.txt`
or
`pip install -r requirements.txt`

## Streamlit app

To run our streamlit app you need to prepare the data.

### download the source file from kaggle

Download ecg dataset from kaggles. They should be named as `mitbih_test.csv` `mitbih_train.csv` `ptbdb_normal.csv` `ptbdb_abnormal.csv`.

Move those files into the folder `data/raw`.

### excute notebooks to generate processed dataset.

run notebooks under `notebooks`  
`preprocessing_ptb_clean.ipynb`  
`preprocessing_mit_clean.ipynb`

### run streamlit app at port 8080

run `streamlit run ./src/streamlit/app.py --server.runOnSave true --server.port 8080`
