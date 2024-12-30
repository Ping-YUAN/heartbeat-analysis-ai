import streamlit as st


def show_page():
    st.header("Data Visualization")

    st.write(
        """
        In this section, we present various visualizations to help understand the dataset and the results of our analysis.
        The figures below have been pre-generated to save computation time and ensure consistent presentation.
    """
    )

    with st.expander("Raw ECG Signals"):
        st.write(
            """ This visualization aids in understanding the variations and patterns in heartbeat data,
                 crucial for developing accurate machine learning models for heartbeat classification."""
        )

        dataset_option_ecg = st.selectbox(
            "Select a dataset to visualize:",
            ("MIT Dataset", "PTB Dataset"),
            key="dataset_option_ecg",
        )

        if dataset_option_ecg == "MIT Dataset":
            st.markdown("##### Figure 1: Raw ECG Signals for MIT Dataset")
            st.image(st.session_state.images.get("mit_ecg_signals"), width=600)
        elif dataset_option_ecg == "PTB Dataset":
            st.markdown(
                "##### Figure 2: Raw normal and abnormal ECG Signals for PTB Dataset"
            )
            st.image(st.session_state.images.get("ptb_ecg_signals"), width=600)

    with st.expander("The distribution of different heartbeat categories"):
        st.write(
            """ This analysis reveals that the majority of the data consists of normal heartbeats,
                 while the other categories are significantly less frequent. 
                 Understanding this distribution is essential for developing and training machine learning models,
                 as it impacts the model’s ability to accurately classify and detect abnormalities in heartbeats."""
        )

        dataset_option_pie = st.selectbox(
            "Select a dataset to visualize:",
            ("MIT Dataset", "PTB Dataset"),
            key="pie",
        )
        if dataset_option_pie == "MIT Dataset":
            st.markdown(
                "##### Figure 1: The distribution of different heartbeat categories for MIT Dataset"
            )
            st.image(st.session_state.images.get("pie_distribution_mit"), width=600)
        elif dataset_option_pie == "PTB Dataset":
            st.markdown(
                "##### Figure 2: The distribution of different heartbeat categories for PTB Dataset"
            )
            st.image(st.session_state.images.get("pie_distribution_ptb"), width=600)

    with st.expander("PCA Dimensionality Reduction"):
        st.write(
            """PCA is a statistical technique used to reduce the dimensionality of a dataset while retaining most of the variation in the data. 
               This can be useful for visualization and for improving the performance of machine learning algorithms."""
        )

        dataset_option_pca = st.selectbox(
            "Select a dataset to visualize PCA:",
            ("MIT Dataset", "PTB Dataset"),
            key="pca",
        )

        if dataset_option_pca == "MIT Dataset":
            st.markdown("##### Scree Plot for MIT Dataset")
            st.image(st.session_state.images.get("mit_screeplot"), width=400)
            st.markdown("##### PCA Visualization for MIT Dataset")
            st.image(st.session_state.images.get("mit_pca"), width=400)

        elif dataset_option_pca == "PTB Dataset":
            st.markdown("##### Scree Plot for PTB Dataset")
            st.image(st.session_state.images.get("ptb_screeplot"), width=400)
            st.markdown("##### PCA Visualization for PTB Dataset")
            st.image(st.session_state.images.get("ptb_pca_3d"), width=400)

    with st.expander("t-SNE Dimensionality Reduction"):
        st.write(
            """t-SNE is a machine learning algorithm for visualization that is particularly well suited for embedding high-dimensional data into a space of two or three dimensions.
               This helps in visualizing the structure and clusters in high-dimensional data."""
        )

        dataset_option_tsne = st.selectbox(
            "Select a dataset to visualize t-SNE:",
            ("MIT Dataset", "PTB Dataset"),
            key="tsne",
        )

        if dataset_option_tsne == "MIT Dataset":
            st.markdown("##### t-SNE Visualization for MIT Dataset")
            st.image(st.session_state.images.get("mit_tsne"), width=600)
        elif dataset_option_tsne == "PTB Dataset":
            st.markdown("##### t-SNE Visualization for PTB Dataset")
            st.image(st.session_state.images.get("ptb_tsne"), width=600)


# show_page()
