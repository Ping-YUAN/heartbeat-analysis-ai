from utils.helper import shift_row
import joblib
import os
from pathlib import Path

class MitModelShiftWrapper:
    def __init__(self, model):
        self.model = model
        mit_shift_scaler = os.path.join( Path(__file__).parent, "mit_shift_scaler.pkl")
        self.scaler = joblib.load(mit_shift_scaler)
        
    def predict(self, data):
        shifted_data = shift_row(data)
        shifted_data = shifted_data.values.reshape(1, -1)
        shifted_data = self.scaler.transform(shifted_data)
        shifted_data_reshaped = shifted_data.reshape(1, 187,1)
        return self.model.predict(shifted_data_reshaped)