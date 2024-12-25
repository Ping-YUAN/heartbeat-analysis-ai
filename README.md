# Heartbeat Analysis

This repo is a project for heartbeat classification.

The datasets we used come from [mit](https://www.physionet.org/physiobank/database/mitdb/) and [ptb](https://www.physionet.org/physiobank/database/ptbdb/).
We trained several model with classical machine learning and deep learning model to classifique the normal and abormal even multiple classification.

You can find our [streamlit app](https://heartbeat-analysis-ai.streamlit.app/) to get more information

## How to run this project

The datasets come from [kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data).

To run this project, you need to download those 4 files, and put them under `data/raw` folder with the original name.

Recommand using conda to management the enviorment,

Recoomand intercepter version is: 3.10.14
`conda install --yes --file requirements.txt`
or
`pip install -r requirements.txt`

Then you can check the files under the folder `notebooks`, there are notebooks for data inspect/preprocessing/modeling/explanation.

## File structure

To understand the file structure check [Project Organization](./organization.md).

### run streamlit app on local at port 8080

To run our streamlit app on local.

run `streamlit run ./src/streamlit/app.py --server.runOnSave true --server.port 8080`
