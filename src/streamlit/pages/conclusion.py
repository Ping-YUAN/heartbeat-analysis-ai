import streamlit as st

def show_page():
    st.header("Conclusion")

    st.subheader("Summary of Findings")
    st.write(
        """
    Our modeling efforts demonstrated strong classification performance across both the MIT and PTB datasets, with the DNN model achieving the highest accuracy. Through systematic hyperparameter tuning, we optimized the KNN, Decision Tree, and XGBoost models, with XGBoost achieving solid performance on the PTB dataset, but less so on MIT.
    """
    )

    st.subheader("Interpretability Analysis")
    st.write(
        """
    Our interpretability analysis, using SHAP and LIME, highlighted critical features influencing predictions, particularly early-cycle features in the heartbeat sequence that proved significant in the DNN’s decision-making. These insights suggest that key segments of the heartbeat carry essential signals for classification, which the models leverage effectively. The local interpretability provided by LIME offered additional context for individual predictions, aligning well with global feature importance in many cases.
    """
    )

    st.subheader("Model Performance and Future Work")
    st.write(
        """
    While the DNN's architecture proved effective for both datasets, further tuning is needed for LSTM and Transformer models due to their unexpectedly low performance. Given differences in feature importance across datasets, merging the MIT and PTB datasets for a combined analysis may enhance model generalizability and robustness. This integration will support a more comprehensive interpretability framework, enabling a deeper understanding of feature contributions across diverse heartbeat signals.
    """
    )

    st.subheader("Project Overview")
    st.write(
        """
    In this report, we use advanced machine learning and deep learning techniques to detect whether a heartbeat is normal or abnormal. It aims to improve the accuracy of heartbeat detection and classification within general supervision by using both traditional machine learning techniques, as well as more advanced deep learning models. We provide a comprehensive overview of the methodologies employed, including data exploration, data preprocessing, feature engineering, model selection, and evaluation metrics.
    """
    )

    st.subheader("Conclusion and Future Directions")
    st.write(
        """
    Overall, our results show promising efficacy of these techniques to improve diagnostic accuracy and possibly help in early diagnosis of cardiac anomalies. Supported by expert mentorship, our group work has resulted in detailed discussions and encouraging outcomes that could serve as groundwork for further research on medical diagnosis.
    """
    )
    
# show_page()