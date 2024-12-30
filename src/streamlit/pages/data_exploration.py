import streamlit as st
import pandas as pd
from pathlib import Path
import os

dataset_name = dict(
    {
        "mit_train": "MIT Train",
        "mit_test": "MIT Test",
        "ptb_normal": "PTB Normal",
        "ptb_abnormal": "PTB Abnormal",
    }
)


# Function to explore selected dataset
def explore_dataset(dataset_name_param):
    st.write(f"### {dataset_name_param} Dataset")
    if dataset_name_param == dataset_name.get("mit_train"):
        st.write("Number of records: 87553")
        st.write("Number of columns: 188")
    elif dataset_name_param == dataset_name.get("mit_test"):
        st.write("Number of records: 21892")
        st.write("Number of columns: 188")
    elif dataset_name_param == dataset_name.get("ptb_normal"):
        st.write("Number of records: 4046")
        st.write("Number of columns: 188")
    elif dataset_name_param == dataset_name.get("ptb_abnormal"):
        st.write("Number of records: 10506")
        st.write("Number of columns: 188")

    st.write("Sample data:")
    sample_data = None

    current_dir = Path(__file__).parent
    # file_path =
    if dataset_name_param == dataset_name.get("mit_train"):
        sample_data = pd.read_csv(
            os.path.join(current_dir, "../sample", "mit_train.csv")
        )
    elif dataset_name_param == dataset_name.get("mit_test"):
        sample_data = pd.read_csv(
            os.path.join(current_dir, "../sample", "mit_test.csv")
        )
    elif dataset_name_param == dataset_name.get("ptb_normal"):
        sample_data = pd.read_csv(
            os.path.join(current_dir, "../sample", "ptb_normal.csv")
        )
    elif dataset_name_param == dataset_name.get("ptb_abnormal"):
        sample_data = pd.read_csv(
            os.path.join(current_dir, "../sample", "ptb_abnormal.csv")
        )
    st.write(sample_data.head())

    show_missing = st.checkbox("Show total number of missing values")
    if show_missing:
        if dataset_name_param == dataset_name.get("mit_train"):
            st.write("Missing values in MIT Train Dataset: 0")
        elif dataset_name_param == dataset_name.get("mit_test"):
            st.write("Missing values in MIT Test Dataset: 0")
        elif dataset_name_param == dataset_name.get("ptb_normal"):
            st.write("Missing values in PTB Normal Dataset: 0")
        elif dataset_name_param == dataset_name.get("ptb_abnormal"):
            st.write("Missing values in PTB Abnormal Dataset: 0")

    show_duplicate = st.checkbox("Show number of duplicated rows")
    if show_duplicate:
        if dataset_name_param == dataset_name.get("mit_train"):
            st.write("Total number of duplicated rows: 0")
        elif dataset_name_param == dataset_name.get("mit_test"):
            st.write("Total number of duplicated rows: 0")
        elif dataset_name_param == dataset_name.get("ptb_normal"):
            st.write("Total number of duplicated rows: 1")
        elif dataset_name_param == dataset_name.get("ptb_abnormal"):
            st.write("Total number of duplicated rows: 6")

    # Summary information based on dataset selection
    st.write("### Dataset Summary")
    if dataset_name_param == "MIT Train" or dataset_name_param == "MIT Test":
        st.write(
            "Each row represents a single heartbeat, and each column from c-0 to c-186 represents the signal values at different time points. c-187 is the target label for the heartbeat class."
        )
    elif dataset_name_param == "PTB Normal":
        st.write(
            "Each row represents a single heartbeat, and each column from c-0 to c-186 represents the signal values at different time points. c-187 is the target label which is 0 for normal heartbeats."
        )
    elif dataset_name_param == "PTB Abnormal":
        st.write(
            "Each row represents a single heartbeat, and each column from c-0 to c-186 represents the signal values at different time points. c-187 is the target label which is 1 for abnormal heartbeats."
        )

    # Show detailed information
    show_details = st.checkbox("Show more detailed information")
    if show_details:
        if dataset_name_param == "MIT Train":
            st.write(
                """
                **Details of MIT Train Dataset:** 
                - Size and Structure: The file is quite large, with a size of 392 MB. It contains 87,554 rows, each representing a single heartbeat. 
                - Columns: There are 188 columns in the file. The first 187 columns represent the signal values at different time points, and the last column is the target label for the heartbeat class. 
                - Classes: The target column categorizes heartbeats into five classes: 
                    0: Normal heartbeats (“Normal”) 
                    1: Supraventricular premature beats (“Supraventricular”) 
                    2: Ventricular escape beats (“Ventricular”) 
                    3: Fusion of ventricular and normal beats (“Fusion”) 
                    4: Unclassified beats (“Unclassifiable”) 
                - Class Imbalance: The dataset is known to be imbalanced, with a significant majority of the heartbeats labeled as normal (class “0”). This imbalance needs to be addressed during the model training process to prevent bias. 
            """
            )

        if dataset_name_param == "MIT Test":
            st.write(
                """
                **Details of MIT Test Dataset:** 
                - Size and Structure: The file is approximately 98.1 MB in size. It contains 21,892 rows, with each row representing a single heartbeat. 
                - Columns: Similar to the training file, there are 188 columns. The first 187 columns represent the individual time points of the ECG signal, and the final column is the class label. 
                - Classes: The class labels are the same as in the training set, with five categories ranging from normal heartbeats to various types of arrhythmias. 
                - Usage: This file is used to evaluate the performance of the model trained on the mitbih_train.csv data. It helps in assessing how well the model generalizes to new, unseen data. 
            """
            )

        if dataset_name_param == "PTB Normal":
            st.write(
                """
                **Details of PTB Normal Dataset:** 
                - The PTB Diagnostic ECG Database utilizes a specialized recorder with 16 input channels, including 14 for ECGs, one for respiration, and one for line voltage. 
                - The recorder handles an input voltage of ±16 mV with a high resolution of 16 bits and a bandwidth of 0-1 kHz.
                - It contains 549 records from 290 subjects, with ages ranging from 17 to 87.
                - Signal Measurement: Each record has 15 signals, including the standard 12 ECG leads plus 3 Frank lead ECGs, all digitized at 1000 samples per second.
                - It contains ECG recordings from healthy subjects. The data is structured similarly to the MITBIH files, with each row representing a single heartbeat and columns representing the signal values at different time points. The last column typically contains the label, which in this case would be 0 indicating a normal heartbeat.
            """
            )

        if dataset_name_param == "PTB Abnormal":
            st.write(
                """
                **Details of PTB Abnormal Dataset:** 
                - The PTB Diagnostic ECG Database utilizes a specialized recorder with 16 input channels, including 14 for ECGs, one for respiration, and one for line voltage. 
                - The recorder handles an input voltage of ±16 mV with a high resolution of 16 bits and a bandwidth of 0-1 kHz.
                - It contains 549 records from 290 subjects, with ages ranging from 17 to 87.
                - Signal Measurement: Each record has 15 signals, including the standard 12 ECG leads plus 3 Frank lead ECGs, all digitized at 1000 samples per second.
                - It includes ECG recordings from patients with various heart conditions. The structure is the same as the ptbdb_normal.csv file, with the last column containing the label 1 to indicate an abnormal heartbeat.
            """
            )


def show_page():
    st.header("Data Exploration")

    # Introduction
    st.subheader("Dataset Overview")
    st.write(
        """
        In this experiment, we utilize the freely available Heartbeat dataset 
        from Kaggle,
        curated by Shayan Fazeli. It includes various ECG signals classified 
        into different
        categories such as 'Normal' and various 'Abnormal' heartbeats.
        
        We work with two ECG Heartbeat Categorization Datasets:
        the MIT-BIH Arrhythmia Database (mitbih) and the PTB Diagnostic ECG 
        Database (ptbdb).
        
        The MIT-BIH Arrhythmia Database contains 48 half-hour ECG recordings 
        from 47 individuals recorded
        between 1975-1979. The recordings were digitized at 360 samples per 
        second with 11-bit resolution.
        
        The PTB Diagnostic ECG Database consists of recordings from healthy 
        subjects and patients
        with various heart conditions. It includes 15 signals for each record, 
        with data digitized at 1000 samples per second.
    """
    )

    # Dataset Selection
    dataset_option = st.selectbox(
        "Select a dataset to explore",
        (
            dataset_name.get("mit_train"),
            dataset_name.get("mit_test"),
            dataset_name.get("ptb_normal"),
            dataset_name.get("ptb_abnormal"),
        ),
    )

    explore_dataset(dataset_option)

    # if dataset_option == dataset_name.get('mit_train'):
    #     explore_dataset(mit_train_df, dataset_option)
    # elif dataset_option ==  dataset_name.get('mit_test'):
    #     explore_dataset(mit_test_df, dataset_option)
    # elif dataset_option == dataset_name.get('ptb_normal'):
    #     explore_dataset(ptb_normal_df, dataset_option)
    # elif dataset_option == dataset_name.get('ptb_abnormal'):
    #     explore_dataset(ptb_abnormal_df, dataset_option)
