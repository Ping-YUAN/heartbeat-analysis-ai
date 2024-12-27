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
'mit_binary_shift_shap': asset_path + "mit_binary_shift_shap.png"
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