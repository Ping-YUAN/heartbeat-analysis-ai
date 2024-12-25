import streamlit as st

def show_page():
    st.header("References")
    st.write(
        """
        Here are the references used in this project:
        1. [GitHub: Heartbeat Analysis AI](https://github.com/Ping-YUAN/heartbeat-analysis-ai)
        2. [Kaggle: Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
        3. Mark RG, Schluter PS, Moody GB, Devlin, PH, Chernoff, D. (1982). An annotated ECG database for evaluating arrhythmia detectors. IEEE Transactions on Biomedical Engineering, 29(8), 600.
        4. Moody GB, Mark RG. (1990). The MIT-BIH Arrhythmia Database on CD-ROM and software for use with it. Computers in Cardiology, 17, 185-188.
        5. Moody GB, Mark RG. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology, 20(3), 45-50. (PMID: 11446209)
        6. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101(23), e215–e220.
        
    """
    )

show_page()