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
        From the dataset description: we get that the ecg sample data is collected with the 125Hz freqency. (125 signals per second)   
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

    st.image(st.session_state.images.get("ecg-model"), width=400)

    st.markdown(
        """
        From the ecg signal chart, we can see that R wave peak is always(almost from the mit dataset) the highest signal.   
        The idea of data shift is to align the R wave peak to a fixed columns. 
        In that case, when we know the important feature, we can compare the feature number with the aligned feature to know which period of the important feature belongs to.
        """
    )

    st.markdown(
        """
        As we only need 75 - 125 features to cover a completed ecg period. We may have several R wave peak in our sample data.   
        We need to exclude the R wave peak which appear at beginning and ending part to ensure that we align a complete ecg period signal to the center that we want.   
        
        The R wave duration around 0.06 - 0.12 seconds, and happend about 40%-50% process of entire heartbeat.   
        If we calcluat it the R wave might appear around [30, 37].  
        We only count the peak in range of [15, 150] to improve the accuracy of the data shift. 
        """
    )
    st.write("### how to choose the fix column for R wave peak")
    st.markdown(
        """
        By applying the algorithm decribed above, we calcuate the average of R wave peak which is **87**.     
        We will therefore shift the R wave peak to the 87st columns(c_86 column name) for training data and transform the test data in the same way before predict.   
        
        *It's not a perfect solution but after applying to the shift data, we get a good enough performance of our model. So we applied this method to our modeling as preprocessing*  
        """
    )
    st.write("## Comparsion models with shift and raw data")
    st.markdown(
        """
        Macro Average f1 score for multiple classification   
        | data type    | XGBoost | KNN  | DNN | LSTM | CNN | 
        |--------------|---------|------|-----|------|-----|
        | raw data     | 0.58    | 0.84 | 0.74| O.67 | 0.74|
        | shifted data | 0.48    | 0.79 | 0.76| 0.70 | 0.75|
        
        Macro Average f1 score for binary classification  
        | data type    | XGBoost | KNN  | DNN | LSTM | CNN | 
        |--------------|---------|------|-----|------|-----|
        | raw data     | 0.85    | 0.93 | 0.93| O.92 | 0.95|
        | shifted data | 0.74    | 0.92 | 0.90| 0.88 | 0.95|


        We can see that in general the performance on shift data is lower thatn the raw data which is normal.   
        The raw data can have naturally data augmentation which can let model learning with many different cases.   
        However, the performance of the model with shifted data doesn't reduce much of there.   
        And we can benefit with the better interpretability, so in the following chapter we will keep showing on model and interpretabilty both global and local based on model with shfited data. 
        """
    )
    st.write("## Potential better solution.")
    st.markdown(
        """
        The way we shift data is not perfect but enough for now.    
        we only get a slight reduction in forecast accuracy.    
        
        However, we found an article from [Nature](https://www.nature.com/articles/s41598-021-97118-5) which is more advanced and profesional approach than us.   
        If we have time we will try to implement the algorithm to have a professional data shift for ecg signal. 
        """
    )
    # explain how to process the data shift and why.
    # add link for a more reasonable way to do that


# show_page()
