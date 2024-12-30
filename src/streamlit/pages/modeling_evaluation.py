import streamlit as st


def show_page():
    st.header("Modeling & Evaluation")
    # only text and image table to demostrate that
    # multiple classification model created and evaluated.
    # which one will be used for demo
    # binary classification model create and evaluated.
    # which one will be used for demo.
    st.markdown(
        """
    The model we present here will be based on the **shifted data**. We also trained based on raw data. 
    As for the f1 score that we are using, since our dataset is **imbalanced** and the abnormal cases will be more important for us, We will use macro f1 score for model evaluation.    
    We will show the conclusion first then following by the supported details in the expandable panel. 
    """
    )

    st.write("## Conclusion of modeling and evaluation")
    st.markdown(
        """
    After applying several models for mutiple classification and binary classification. 
    Our conclusion is:  
    * 1. Multipe classification doesn't work well. We won't recommand to apply multiple classification model.   
    * 2. Shifted data works well for modeling, accuracy looks reliable and trustable.   
    * 3. CNN model works best in binary classification with macro average f1 score: 0.95.  
    * 4. Global interpretability with Shap for selected CNN model looks reasonable.(Almost all the top features impact on model output are around c_86 as we shift the R wave peak to 87 column whose name is c_86)
    """
    )

    show_cnn_global_interpretability = st.checkbox(
        "Show CNN model important features for model ouputs"
    )
    if show_cnn_global_interpretability:
        st.image(
            st.session_state.images.get("mit_binary_shift_shap"),
            use_container_width=True,
        )

    with st.expander("Multiple classification"):
        # Show XGBoost multiple Model
        show_xgboost_multiple = st.checkbox("Show XGBoost multiple Model")
        if show_xgboost_multiple:
            st.write("## XGBoost")
            st.image(st.session_state.images.get("mit-xgb-multiple-confusion-matrix"))
            st.write("Classification report over test set")

            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.76   | 0.86     | 18118   |
        | Supraventricular  | 0.20      | 0.71   | 0.32     | 556     |
        | Ventricular       | 0.40      | 0.85   | 0.55     | 1448    |
        | Fusion            | 0.11      | 0.88   | 0.20     | 162     |
        | **Accuracy**      |           |        | 0.77     | 20284   |
        | **Macro Avg**     | 0.43      | 0.8    | 0.48     | 20284   |
        | **Weighted Avg**  | 0.92      | 0.77   | 0.82     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.82 which looks good.    
        However because of the extremely imbalanced dataset, the result get hightly affected by Normal class. 
        Where as when we check the Macro average f1 score, it only has 0.48 which is not idea.
        """
            )
        # Show KNN multiple Model
        show_knn_multiple = st.checkbox("Show KNN multiple Model")
        if show_knn_multiple:
            st.write("## KNN")
            st.image(st.session_state.images.get("mit-knn-multiple-confusion-matrix"))
            st.write("Classification report over test set")

            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.90      | 0.97   | 0.98     | 18118   |
        | Supraventricular  | 0.61      | 0.71   | 0.65     | 556     |
        | Ventricular       | 0.88      | 0.92   | 0.90     | 1448    |
        | Fusion            | 0.52      | 0.70   | 0.60     | 162     |
        | **Accuracy**      |           |        | 0.96     | 20284   |
        | **Macro Avg**     | 0.75      | 0.83   | 0.78     | 20284   |
        | **Weighted Avg**  | 0.96      | 0.96   | 0.96     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.96 which looks good enough.    
        Even the Macro average f1 score we get 0.78 much much better than XGBoost. 
        It is not that idea but a very good improvement compare to XGBoost. 
        From the result, we can see that, the "Supraventricular" and "Fusion" has general bad performance than other. 
        We can see that the number of sample is smaller compare to the two other cases. Even the ratio generate from test set. It's the same ratio on train set. 
        We may get better performance if we get more sample on those two cases.( "Supraventricular" and "Fusion")
        """
            )
        show_dnn_multiple = st.checkbox("Show DNN multiple Model")
        if show_dnn_multiple:
            st.write("## DNN")

            show_dnn_multiple_architecture = st.checkbox(
                "Show DNN multiple architecture"
            )
            if show_dnn_multiple_architecture:
                st.image(st.session_state.images.get("mit-dnn-architecture"))

            show_dnn_multiple_model_train = st.checkbox("show DNN multiple model train")
            if show_dnn_multiple_model_train:
                st.image(
                    st.session_state.images.get("mit-dnn-multiple-model-loss-by-epoch")
                )
                st.image(
                    st.session_state.images.get(
                        "mit-dnn-multiple-model-accuracy-by-epoch"
                    )
                )
            st.image(st.session_state.images.get("mit-dnn-multiple-confusion-matrix"))

            st.write("Classification report over test set")
            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.96   | 0.97     | 18118   |
        | Supraventricular  | 0.56      | 0.79   | 0.65     | 556     |
        | Ventricular       | 0.83      | 0.92   | 0.88     | 1448    |
        | Fusion            | 0.38      | 0.82   | 0.52     | 162     |
        | **Accuracy**      |           |        | 0.95     | 20284   |
        | **Macro Avg**     | 0.69      | 0.87   | 0.76     | 20284   |
        | **Weighted Avg**  | 0.96      | 0.95   | 0.96     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.96 which looks good enough as KNN.    
        Even the Macro average f1 score we get 0.76 much better than XGBoost similar to KNN. 
        It perform similar as KNN 
        """
            )

        show_lstm_multiple = st.checkbox("Show LSTM multiple Model")
        if show_lstm_multiple:
            st.write("## LSTM")
            show_lstm_multiple_architecture = st.checkbox(
                "Show LSTM multiple architecture"
            )
            if show_lstm_multiple_architecture:
                st.image(st.session_state.images.get("mit-lstm-architecture"))

            show_lstm_multiple_model_train = st.checkbox(
                "show LSTM multiple model train"
            )
            if show_lstm_multiple_model_train:
                st.image(
                    st.session_state.images.get("mit-lstm-multiple-model-loss-by-epoch")
                )
                st.image(
                    st.session_state.images.get(
                        "mit-lstm-multiple-model-accuracy-by-epoch"
                    )
                )
            st.image(st.session_state.images.get("mit-lstm-multiple-confusion-matrix"))

            st.write("Classification report over test set")
            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.91   | 0.95     | 18118   |
        | Supraventricular  | 0.47      | 0.85   | 0.61     | 556     |
        | Ventricular       | 0.59      | 0.93   | 0.72     | 1448    |
        | Fusion            | 0.37      | 0.88   | 0.52     | 162     |
        | **Accuracy**      |           |        | 0.91     | 20284   |
        | **Macro Avg**     | 0.61      | 0.90   | 0.70     | 20284   |
        | **Weighted Avg**  | 0.95      | 0.91   | 0.92     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.92 which looks good enough but lower than KNN and DNN.    
        The Macro average f1 score we get 0.70 much better than XGBoost similar to KNN and DNN. 
        It perform similar as KNN and DNN but a slightly both on Macro averge f1 or weighted average f1 score.
        """
            )

        show_cnn_multiple = st.checkbox("Show CNN multiple Model")
        if show_cnn_multiple:
            st.write("## LSTM")
            show_cnn_multiple_architecture = st.checkbox(
                "Show CNN multiple architecture"
            )
            if show_cnn_multiple_architecture:
                st.image(st.session_state.images.get("mit-cnn-architecture"))

            show_cnn_multiple_model_train = st.checkbox("show CNN multiplemodel train")
            if show_cnn_multiple_model_train:
                st.image(
                    st.session_state.images.get("mit-cnn-multiple-model-loss-by-epoch")
                )
                st.image(
                    st.session_state.images.get(
                        "mit-cnn-multiple-model-accuracy-by-epoch"
                    )
                )
            st.image(st.session_state.images.get("mit-cnn-multiple-confusion-matrix"))

            st.write("Classification report over test set")
            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 1.00      | 0.95   | 0.97     | 18118   |
        | Supraventricular  | 0.61      | 0.89   | 0.73     | 556     |
        | Ventricular       | 0.84      | 0.95   | 0.89     | 1448    |
        | Fusion            | 0.26      | 0.92   | 0.40     | 162     |
        | **Accuracy**      |           |        | 0.95     | 20284   |
        | **Macro Avg**     | 0.68      | 0.93   | 0.75     | 20284   |
        | **Weighted Avg**  | 0.97      | 0.95   | 0.95     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.95 which looks good enough similar than KNN and DNN.    
        The Macro average f1 score we get 0.75 much better than XGBoost similar to KNN and DNN. 
        It perform similar as KNN and DNN.
        """
            )
        st.markdown(
            """
      After perform several model for multiple classfication, we notice that, both model doesn't detect well on the multiple classes especially the cases for "Supraventricular" "Fusion".    
      The reason might be we don't have enough data for those two cases to be well classified.    
      As we also perform the multiple classification on the raw data without shift, the performance is really similar to the shifted model, we can conclude that the weak performance on those cases is not because of the shift.    
      And it also prove that our algorithm in general works well.   
       
      As the multiple classification doesn't work that well, we will focus on the binary classification to provide a reliable model.   
      """
        )

    with st.expander("Binary classification"):
        # Show XGBoost binary Model
        show_xgboost_binary = st.checkbox("Show XGBoost binary Model")
        if show_xgboost_binary:
            st.write("## XGBoost")
            st.image(st.session_state.images.get("mit-xgb-binary-confusion-matrix"))
            st.write("Classification report over test set")

            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.86   | 0.92     | 18118   |
        | Abnormal          | 0.42      | 0.86   | 0.57     | 2166    |
        | **Accuracy**      |           |        | 0.86     | 20284   |
        | **Macro Avg**     | 0.70      | 0.86   | 0.74     | 20284   |
        | **Weighted Avg**  | 0.92      | 0.86   | 0.88     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.88 which looks good.    
        However because of the extremely imbalanced dataset, the result get hightly affected by Normal class. 
        Where as when we check the Macro average f1 score, it decrease to 0.74 which is not idea for binary classification.
        """
            )
        # Show KNN binary Model
        show_knn_binary = st.checkbox("Show KNN binary Model")
        if show_knn_binary:
            st.write("## KNN")
            st.image(st.session_state.images.get("mit-knn-binary-confusion-matrix"))
            st.write("Classification report over test set")

            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.97   | 0.98     | 18118   |
        | Abnormal          | 0.81      | 0.90   | 0.85     | 2166    |
        | **Accuracy**      |           |        | 0.97     | 20284   |
        | **Macro Avg**     | 0.90      | 0.94   | 0.92     | 20284   |
        | **Weighted Avg**  | 0.97      | 0.97   | 0.97     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.97 which looks good enough.    
        The Macro average f1 score we get 0.92 much much better than XGBoost. 
        If we only check for f1 score the KNN model almost works well enough. 
        However, as we said, we care more about the abnormal cases, we would like to see improment over the abnormal f1 score. 
        """
            )
        show_dnn_binary = st.checkbox("Show DNN binary Model")
        if show_dnn_binary:
            st.write("## DNN")

            show_dnn_binary_architecture = st.checkbox(
                "Show DNN binary classification architecture"
            )
            if show_dnn_binary_architecture:
                st.image(st.session_state.images.get("mit-dnn-architecture"))

            show_dnn_binary_model_train = st.checkbox("show DNN binary model train")
            if show_dnn_binary_model_train:
                st.image(
                    st.session_state.images.get("mit-dnn-binary-model-loss-by-epoch")
                )
                st.image(
                    st.session_state.images.get(
                        "mit-dnn-binary-model-accuracy-by-epoch"
                    )
                )
            st.image(st.session_state.images.get("mit-dnn-binary-confusion-matrix"))

            st.write("Classification report over test set")
            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.96   | 0.98     | 18118   |
        | Abnormal          | 0.76      | 0.92   | 0.83     | 2166    |
        | **Accuracy**      |           |        | 0.96     | 20284   |
        | **Macro Avg**     | 0.87      | 0.94   | 0.90     | 20284   |
        | **Weighted Avg**  | 0.97      | 0.96   | 0.96     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.96 which looks good enough as KNN.    
        Even the Macro average f1 score we get 0.90 much better than XGBoost similar to KNN. 
        It perform similar as KNN.
        """
            )

        show_lstm_binary = st.checkbox("Show LSTM binary Model")
        if show_lstm_binary:
            st.write("## LSTM")
            show_lstm_binary_architecture = st.checkbox(
                "Show LSTM binary classification architecture"
            )
            if show_lstm_binary_architecture:
                st.image(st.session_state.images.get("mit-lstm-architecture"))

            show_lstm_binary_model_train = st.checkbox("show LSTM binary model train")
            if show_lstm_binary_model_train:
                st.image(
                    st.session_state.images.get("mit-lstm-binary-model-loss-by-epoch")
                )
                st.image(
                    st.session_state.images.get(
                        "mit-lstm-binary-model-accuracy-by-epoch"
                    )
                )
            st.image(st.session_state.images.get("mit-lstm-binary-confusion-matrix"))

            st.write("Classification report over test set")
            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.95   | 0.97     | 18118   |
        | Abnormal          | 0.69      | 0.94   | 0.80     | 2166    |
        | **Accuracy**      |           |        | 0.95     | 20284   |
        | **Macro Avg**     | 0.84      | 0.95   | 0.88     | 20284   |
        | **Weighted Avg**  | 0.96      | 0.95   | 0.95     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.95 which looks good enough similar with KNN and DNN.    
        The Macro average f1 score we get 0.88 much better than XGBoost similar to KNN and DNN. 
        It perform similar as KNN and DNN but a slightly both on Macro averge f1 or weighted average f1 score.
        """
            )

        show_cnn_binary = st.checkbox("Show CNN binary Model")
        if show_cnn_binary:
            st.write("## LSTM")
            show_cnn_binary_architecture = st.checkbox("Show CNN binary architecture")
            if show_cnn_binary_architecture:
                st.image(st.session_state.images.get("mit-cnn-architecture"))

            show_cnn_binary_model_train = st.checkbox("show CNN binary model train")
            if show_cnn_binary_model_train:
                st.image(
                    st.session_state.images.get("mit-cnn-binary-model-loss-by-epoch")
                )
                st.image(
                    st.session_state.images.get(
                        "mit-cnn-binary-model-accuracy-by-epoch"
                    )
                )
            st.image(st.session_state.images.get("mit-cnn-binary-confusion-matrix"))

            st.write("Classification report over test set")
            st.markdown(
                """
        | Class             | Precision | Recall | F1-Score | Support |
        |-------------------|-----------|--------|----------|---------|
        | Normal            | 0.99      | 0.98   | 0.99     | 18118   |
        | Abnormal          | 0.87      | 0.95   | 0.91     | 2166    |
        | **Accuracy**      |           |        | 0.98     | 20284   |
        | **Macro Avg**     | 0.93      | 0.96   | 0.95     | 20284   |
        | **Weighted Avg**  | 0.98      | 0.98   | 0.98     | 20284   | 
        """
            )
            st.markdown(
                """
        The weighted average f1 score we get 0.98 which looks perfect even slight improvement than KNN, DNN and LSTM.    
        The Macro average f1 score we get 0.95 much better than XGBoost, KNN, DNN and LSTM. 
        The CNN looks perform the best between those model that we have even trained. 
        """
            )
        st.markdown(
            """
      After perform several model for binary classfication, we notice that, both model detect well on the binary classes.
      As we also perform the binay classification on the raw data without shift, the performance is really similar to the shifted model.
      We can conclude that our CNN model with shift data in general works well.   
      """
        )
