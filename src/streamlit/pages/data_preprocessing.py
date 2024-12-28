import streamlit as st

def show_page():
    st.header("Data Preprocessing")

    st.write(
        """
        Starting from this section, we focus on the MIT dataset for our data preprocessing/modeling/demo.
        We have chosen to focus on the MIT dataset due to its comprehensive representation and relevance to our study.
        """
    )

    with st.expander("Handling Missing Values"):
        st.write(
            """
            Handling missing values is a critical step in data preprocessing.
            However, in this case, we are fortunate to have no missing values in the MIT dataset.
            Therefore, no imputation or removal of missing values is required at this stage. 
            """
        )

        show_missing_values = st.checkbox("Show missing values")

        if show_missing_values:
            st.write(f"- MIT Train Dataset missing values: {0}")
            st.write(f"- MIT Test Dataset missing values: {0}")

    with st.expander("Data Cleaning"):
        st.write(
            """
            Data cleaning involves correcting or removing incorrect, corrupted, or irrelevant parts of the dataset. Steps include:
            """
        )

        show_duplicate = st.checkbox("Show duplicate")
        if show_duplicate:
            st.write(f"- MIT Train Dataset duplicate rows: {0}")
            st.write(f"- MIT Test Dataset duplicate rows: {0}")

        st.write(
            """
            - Removing duplicate rows to ensure each row represents unique information.
            """
        )

    with st.expander("Split the Data"):
        st.write(
            """
            Before applying any resampling and rescaling techniques, we split the MIT dataset into training and testing sets. 
            This prevents any data leakage and ensures that the model is evaluated on unseen data.
            """
        )
        # Example code for splitting the data
        # st.write(f"MIT Training set size: {X_train_mit.shape[0]}")
        # st.write(f"MIT Test set size: {X_test_mit.shape[0]}")
    with st.expander("Rescaling Results"):
        st.write(
            """
            Below are the mean F1-scores for different combinations of rescaling methods and models applied to the MIT dataset.
            """
        )

        # Creating the table for rescaling results with highlighted cell
        st.markdown(
            """
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>StandardScaler</th>
                        <th>MinMaxScaler</th>
                        <th>RobustScaler</th>
                        <th>None</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>LogisticRegression</td>
                        <td>0.408</td>
                        <td>0.408</td>
                        <td>0.409</td>
                        <td>0.408</td>
                    </tr>
                    <tr>
                        <td>Decision Tree</td>
                        <td>0.781</td>
                        <td>0.781</td>
                        <td>0.781</td>
                        <td>0.781</td>
                    </tr>
                    <tr>
                        <td>KNN</td>
                        <td>0.847</td>
                        <td class="highlight">0.855</td>
                        <td>0.846</td>
                        <td>0.854</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True
        )

        st.write(
            """
            **Benefits of Rescaling:**
            - Rescaling methods such as StandardScaler, MinMaxScaler, and RobustScaler ensure that features have comparable scales, which helps improve the performance and convergence rate of the models.
            - It is generally advisable to apply rescaling methods after resampling to avoid introducing bias during the scaling process.

            **Note:** After testing various combinations, we have found that MinMaxScaler works best for our dataset.
            """
        )
        
    with st.expander("Resampling Results"):
        st.write(
            """
            Below are the mean F1-scores for different combinations of resampling methods and models applied to the MIT dataset.
            """
        )

        # Creating the table for resampling results with highlighted cell
        st.markdown(
            """
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>SMOTE</th>
                        <th>OverSampling</th>
                        <th>UnderSampling</th>
                        <th>None</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>LogisticRegression</td>
                        <td>0.408</td>
                        <td>0.408</td>
                        <td>0.407</td>
                        <td>0.408</td>
                    </tr>
                    <tr>
                        <td>Decision Tree</td>
                        <td>0.760</td>
                        <td>0.788</td>
                        <td>0.626</td>
                        <td>0.781</td>
                    </tr>
                    <tr>
                        <td>KNN</td>
                        <td>0.824</td>
                        <td class="highlight">0.835</td>
                        <td>0.752</td>
                        <td>0.835</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True
        )

        st.write(
            """
            **Benefits of Resampling:**
            - Resampling methods such as SMOTE, Oversampling, and Undersampling help address class imbalance in the dataset, which can improve the model's ability to generalize and avoid bias towards the majority class.
            - It is generally advisable to apply resampling methods before rescaling to avoid introducing bias during the scaling process.

            **Note:** After testing various combinations, we have found that Oversampling works best for our dataset.
            """
        )


# show_page()