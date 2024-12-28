
import lime.lime_tabular
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tensorflow.keras.models import load_model
import sys
import lime
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_shift_wrapper import MitModelShiftWrapper 
from utils.helper import shift_row

current_dir = Path(__file__).parent

all_class_mapping = {
  0: 'Normal',
  1: 'Supraventricular',
  2: 'Ventricular',
  3: 'Fusion',
}

@st.cache_resource()
def load_binary_model():
  model_path = os.path.join(current_dir, "../utils", "model_mit_binary_shift_cnn.h5")
  loaded_model = load_model(model_path)
  # Compile the model (ensuring metrics match expected inputs)
  loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
  return loaded_model

# load model 
binary_model = load_binary_model()

@st.cache_resource()
def load_test_data():
  test_data_path = os.path.join(current_dir, "../sample", "mit_test.csv")
  return pd.read_csv(test_data_path, header=0)

@st.cache_resource()
def load_train_data():
  train_data_path = os.path.join(current_dir, "../sample", "mit_train.csv")
  train_df = pd.read_csv(train_data_path, header=0)
  train_df['target'] = train_df['target'].replace([1, 2, 3], 1)
  return train_df

mit_train_data = load_train_data()
mit_test_data = load_test_data()


def getHeartBeatReadableClass(target):
    return all_class_mapping.get(target)
  
def predict_proba_wrapper(data):
  predictions = binary_model.predict(data.reshape(-1, 187, 1))
  return np.column_stack([1 - predictions, predictions])
  
def show_model_interpretability(data, y):
  X_train = mit_train_data.head().drop('target', axis=1)
  y_train = mit_train_data.head()['target']
  explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    training_labels=np.array(y_train),
    mode="classification",  # Set to 'classification' for binary classification
    feature_names=X_train.columns,  # Feature names
    class_names=['Normal', 'Abnormal'],  # Class names for binary classification
    discretize_continuous=True  # Discretizes continuous features
  )
  instance = data.values.reshape(1, 187,1).reshape(-1)
  exp = explainer.explain_instance(
    data_row= instance,
    predict_fn= predict_proba_wrapper,
    num_features=10
  )
  
  feature_imporant = exp.as_list()
  
  lime_df = pd.DataFrame(feature_imporant, columns=['Feature', 'Contribution'])
  fig, ax = plt.subplots()
  sns.barplot(
      x='Contribution', 
      y='Feature', 
      data=lime_df, 
      palette='viridis', 
      orient='h'
  )
  ax.set_title(f'LIME Explanation for Instance ( True Label: {y})')
  ax.set_xlabel('Feature Contribution')
  ax.set_ylabel('Feature')
  st.pyplot(fig)  
  
def show_page():    
    st.header("Modeling Demo & Interpretability")
    st.markdown(
      """
      Based on the modeling part, we will demo the model targeted to **binary classification** with the **data shift** applied. 
      You can find the visualization for raw and shifted below for a given sample.   
      Then you will get the prediction with the model we have trained with **Convolutional Nerural Netowork**   
      Finaly you can check the **interpretability of the sample data** that you choosed.   
      (ps: As data shift is to align R peak around to c_87 column,   
      you can check the important feature based on the shift chart and the Morphology of a normal ECG to know which period may lead to the model predict as abnormal. )
      """)
    st.write('## test data overview ')
    columns = list(mit_test_data.columns)
    data_reordered = mit_test_data[[columns[-1]] + columns[:-1]]
    data_reordered['target'] = data_reordered['target'].map(all_class_mapping)
    # Show raw ecg signal in table
    show_ecg_raw_table = st.checkbox("Show ECG raw signal in table")
    if show_ecg_raw_table:
      st.dataframe(data_reordered)
    selected_data = st.selectbox(
        "Select an sample data:",
        options= [ f"{mit_test_data_idx}-{ 'Normal' if mit_test_data.iloc[mit_test_data_idx]['target'] < 1 else 'Abnormal'}-{ getHeartBeatReadableClass( mit_test_data.iloc[mit_test_data_idx]['target'])}"  for mit_test_data_idx in  mit_test_data.index.tolist()  ]  # Convert index to list for the dropdown
    )
    
    selected_index = int(selected_data.split("-")[0])

    if selected_index > -1:
      selected_row = mit_test_data.loc[selected_index]  # Retrieve the corresponding row
      st.write(f"### Selected Heartbeat type: {'Abnormal' if selected_row['target'] > 0 else 'Normal'} ({getHeartBeatReadableClass(selected_row['target'])})")
      mitModel = MitModelShiftWrapper(binary_model)
      single_ecg = mit_test_data.iloc[selected_index]
      X_test = single_ecg.drop('target')
      y_test = single_ecg['target']
      X_test_shifted = shift_row(X_test)
      # Show raw signal in chart
      show_raw_plot = st.checkbox("Show raw ECG signal in chart")
      if show_raw_plot: 
        fig, ax = plt.subplots()
        ax.plot(mit_test_data.columns[: len(mit_test_data.columns) - 1], X_test)
        ax.set_xticklabels([])
        ax.set_xlabel( all_class_mapping.get(selected_row['target']) )
        ax.set_ylabel("ECG signal 125hz")
        st.pyplot(fig)
      # Show shifted signal in chart
      show_shifted_plot = st.checkbox("Show shifted ECG signal in chart")
      if show_shifted_plot:
        fig, ax = plt.subplots()
        ax.plot(mit_test_data.columns[: len(mit_test_data.columns) - 1], X_test_shifted)
        ax.set_xticklabels([])
        ax.set_xlabel( all_class_mapping.get(selected_row['target']) )
        ax.set_ylabel("ECG signal 125hz")
        st.pyplot(fig)
      # show normal ecg signal
      show_normal_ecg = st.checkbox('Show normal ecg signal')
      if show_normal_ecg:
         st.image(st.session_state.images.get('ecg-model'), width=400)
      predict_result = mitModel.predict(X_test)[0, 0]
      predict_rounded_probability = "{:.2f}".format(round(predict_result * 100, 2) if predict_result > 0.5 else round((1 - predict_result)* 100, 2))
      st.write(f"predicted result: {'Abnormal' if predict_result > 0.5 else 'Normal' } ({ predict_rounded_probability }%)")
      st.write('real result:', 'Abornmal' if y_test > 0 else 'Normal')
    
      # Show interpretability chart 
      show_interpretability_plot = st.checkbox("Show interpretability of sample data")
      if show_interpretability_plot : 
        st.write('## Interpretability of the data ')  
        show_model_interpretability(X_test, y_test)

# show_page()