import streamlit as st

def show_page(): 
    st.header("Project Overview")
    # Layout to place images at the corner
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Introduction")
        st.write(
            """
        This project aims to detect the normality and abnormality of heartbeat 
        rates using advanced machine and deep learning techniques.
        Understanding and accurately identifying abnormal heartbeats can play 
        a crucial role in diagnosing potential heart conditions early.
        """
        )

        st.subheader("Objective")
        st.write(
            """
        The main goal of this project is to develop and evaluate models that 
        can classify heartbeat data as normal or abnormal.
        Users will gain insights into the methods used and the performance of 
        different models.
        """
        )

        st.subheader("Data Source")
        st.write(
            """
        The dataset used for this project is sourced from [this Kaggle dataset]
        (https://www.kaggle.com/datasets/shayanfazeli/heartbeat). 
        It includes comprehensive ECG data and has been preprocessed for
        analysis.
        """
        )

        st.subheader("Techniques Used")
        st.write(
            """
        We employed various machine and deep learning techniques, including 
        logistic regression, decision trees, and neural networks.
        """
        )

        st.subheader("Project Structure")
        st.write(
            """
        The app is divided into several sections:
        - **Data Exploration**: Explore the dataset and key statistics.
        - **Data Preprocessing**: Details on data cleaning and preparation.
        - **Feature Engineering**: Techniques used for feature selection and 
        creation.
        - **Data Visualization**: Visual representation of the data.
        - **Modeling & Evaluation**: Building and assessing the models.
        - **Model Comparison**: Comparing the performance of different models.
        - **Conclusion**: Key findings and future work.
        - **User Guide**: Instructions for using the app.
        - **References**: Sources and references used in the project.
        """
        )

    with col2:
        st.image(st.session_state.images.get('medical-cardio'), use_container_width=True)
        st.image(st.session_state.images.get('ecg'), use_container_width=True)

# show_page()