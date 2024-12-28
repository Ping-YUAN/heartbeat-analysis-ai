import streamlit as st

# from pages import project_overview, data_exploration
st.markdown(
    """ 
    <style>
    
    [data-testid="stSidebarNav"] {
        display: none
    }
      .highlight {
        background-color: #32CD32;
    }
    </style>
    """, 
    unsafe_allow_html=True,
)

images = dict()

def is_community_cloud():
    
    # Check if specific environment variable exists
    # st.write('secrets', st.json(st.secrets))
    if st.secrets is None:
        return False
    
    return  st.secrets and st.secrets.get("ENV", None) is not None

asset_path = 'asset/'
if is_community_cloud():
    asset_path = "https://raw.githubusercontent.com/Ping-YUAN/heartbeat-analysis-ai/main/asset/"
else:
    asset_path = 'asset/'

images = dict({
'medical-cardio': asset_path + "medical-cardio.jpg",
'ecg': asset_path + "ECG.jpg",
'ecg-model': asset_path + 'ecg-model.png',
'mit_ecg_signals': asset_path + "MIT_ECG_Signals.png",
'ptb_ecg_signals': asset_path + "PTB_ECG_Signals.png",
'pie_distribution_mit': asset_path + "pie_distribution_mit.png",
'pie_distribution_ptb': asset_path + "pie_distribution_ptb.png",
'mit_screeplot': asset_path + "MIT_screeplot.png",
'ptb_tsne': asset_path + "PTB_tsne.png",
'mit_tsne': asset_path + "MIT_tsne.png",
'ptb_pca_3d': asset_path + "PTB_PCA_3d.png",
'ptb_screeplot': asset_path + "PTB_screeplot.png",
'mit_pca': asset_path + "MIT_PCA.png",
'mit_binary_shift_shap': asset_path + "mit_binary_shift_shap.png",
'mit-dnn-architecture': asset_path + "mit-dnn-architecture.png",
'mit-cnn-architecture': asset_path + "mit-cnn-architecture.png",
'mit-lstm-architecture': asset_path + "mit-lstm-architecture.png",
'mit-xgb-multiple-confusion-matrix': asset_path + 'mit-xgb-multiple-confusion-matrix.png',
'mit-xgb-binary-confusion-matrix': asset_path + 'mit-xgb-binary-confusion-matrix.png',
'mit-knn-multiple-confusion-matrix': asset_path + 'mit-knn-multiple-confusion-matrix.png',
'mit-knn-binary-confusion-matrix': asset_path + 'mit-knn-binary-confusion-matrix.png',
'mit-dnn-multiple-confusion-matrix': asset_path + 'mit-dnn-multiple-confusion-matrix.png',
'mit-dnn-binary-confusion-matrix': asset_path + 'mit-dnn-binary-confusion-matrix.png',
'mit-dnn-multiple-model-loss-by-epoch': asset_path + 'mit-dnn-multiple-model-loss-by-epoch.png',
'mit-dnn-multiple-model-accuracy-by-epoch': asset_path + 'mit-dnn-multiple-model-accuracy-by-epoch.png',
'mit-dnn-binary-model-loss-by-epoch': asset_path + 'mit-dnn-binary-model-loss-by-epoch.png',
'mit-dnn-binary-model-accuracy-by-epoch': asset_path + 'mit-dnn-binary-model-accuracy-by-epoch.png',
'mit-lstm-multiple-model-loss-by-epoch': asset_path + 'mit-lstm-multiple-model-loss-by-epoch.png',
'mit-lstm-multiple-model-accuracy-by-epoch': asset_path + 'mit-lstm-multiple-model-accuracy-by-epoch.png',
'mit-lstm-binary-model-loss-by-epoch': asset_path + 'mit-lstm-binary-model-loss-by-epoch.png',
'mit-lstm-binary-model-accuracy-by-epoch': asset_path + 'mit-lstm-binary-model-accuracy-by-epoch.png',
'mit-lstm-multiple-confusion-matrix': asset_path + 'mit-lstm-multiple-confusion-matrix.png',
'mit-lstm-binary-confusion-matrix': asset_path + 'mit-lstm-binary-confusion-matrix.png',
'mit-cnn-multiple-model-loss-by-epoch': asset_path + 'mit-cnn-multiple-model-loss-by-epoch.png',
'mit-cnn-multiple-model-accuracy-by-epoch': asset_path + 'mit-cnn-multiple-model-accuracy-by-epoch.png',
'mit-cnn-multiple-confusion-matrix': asset_path + 'mit-cnn-multiple-confusion-matrix.png',
'mit-cnn-binary-model-loss-by-epoch': asset_path + 'mit-cnn-binary-model-loss-by-epoch.png',
'mit-cnn-binary-model-accuracy-by-epoch': asset_path + 'mit-cnn-binary-model-accuracy-by-epoch.png',
'mit-cnn-binary-confusion-matrix': asset_path + 'mit-cnn-binary-confusion-matrix.png',

})

st.session_state.images = images


# Title of the app
st.title("Deep Learning for Heartbeat Analysis: Normal vs. Abnormal")

# Sidebar for navigation
st.sidebar.title("Table of Contents")
pages = [
    "Project Overview",
    "Data Exploration",
    "Data Visualization",
    "Data Preprocessing",
    "Data Shift",
    "Modeling & Evaluation",
    "Modeling Demo & Interpretability",
    "Conclusion",
    "References",
]
page = st.sidebar.radio("Go to", pages)

if page == "Project Overview":
    # project_overview.show_page()
    from pages import project_overview
    project_overview.show_page()

if page == "Data Exploration":
    from pages import data_exploration
    data_exploration.show_page()

if page == "Data Visualization":
    from pages import data_visualization
    data_visualization.show_page()
if page == "Data Preprocessing":
    from pages import data_preprocessing
    data_preprocessing.show_page()

if page == "Data Shift":
    from pages import data_shift
    data_shift.show_page()

if page == "Modeling & Evaluation":
    from pages import modeling_evaluation
    modeling_evaluation.show_page()

if page == "Modeling Demo & Interpretability":
    from pages import modeling_demo_interpretability
    modeling_demo_interpretability.show_page()
    
if page == "Conclusion":
    from pages import conclusion
    conclusion.show_page()

if page == "References":
    from pages import reference
    reference.show_page()