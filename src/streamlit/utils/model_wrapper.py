
#     all_class_mapping = {
#     0: 'Normal',
#     1: 'Supraventricular',
#     2: 'Ventricular',
#     3: 'Fusion',
#     4: 'Unclassifiable'
# }

class MitModelWrapper: 
    def __init__(self, model):
        self.model = model
    def predict(self,data):
        return self.model.predict(data)
    