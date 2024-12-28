import streamlit as st

def show_page():
    st.header("Data Shift")
    st.write("## Why we want to shift data during preprocessing")
    st.markdown(
        """
        Machine learning project may give us a very perfect accuracy in some cases, which looks good.
        However, with only accuracy is not enough, especially in the anomaly detection cases, where we want to know why.     
        Use cases like:    
        * Healthcare: to help docker understand why algorithm suggest a particular diagnoise.    
        * Finance:    
            * 1. Loan service to explain to borrowers why they were approved or denied   
            * 2. Fraud detection: identify the factors that trigger a fraud alert   
        * Autonomous Vehicles: safety critical decisons    
        * Legal(risk assessment)/Customer Service(churn prediction)  etc...   
        """
    )
    
    st.markdown(
        """
        Heartbeat analysis is in the **healthcare area**.   
        We want to help doctor build trust by providing insights into the model's reasoning.    
        Also, in the real life, docker need to work alongside the model and active as a crucial role to make decision.   
        So we want to find a way to improve the interpretability of our model.    
        
        Here it comes the idea **data shift**: to align the features in the similar shape, then we can check the interpretability with lime.     
        """
        )
    st.write("## How we process it ")
    
    st.markdown(
        """
        From the dataset description: we get that the ecg sample data is collected with the 125Hz freqency.    
        And a normal heartbeat rate(heart rate) from 60 to 100 beats per minute(bpm).    
        So we can calculate the number of ecg samples for one heartbeat by:   
      
        """
    )
    st.latex(r"N= frequency · \frac{60}{heart rate}")
    
    st.markdown(
        """
          We can conclude: to cover a complete heartbeat period we may need ecg signals samples in [75, 125]. 
        """
    )
    
    st.image(st.session_state.images.get('ecg-model'), width=400)
    
    st.write("## Potential better solution.")
    st.markdown(
        """
        The way we shift data is not perfect but enough for now.    
        Even thought, in the dataset of mit it looks good, we only get a slight reduction in forecast accuracy.    
        
        However, we found an article from [Nature](https://www.nature.com/articles/s41598-021-97118-5) which is more advanced and profesional approach than us.   
        If we have time we will try to implement the algorithm to have a professional data shift for ecg signal. 
        """
    )
    # explain how to process the data shift and why. 
    # add link for a more reasonable way to do that
# show_page()