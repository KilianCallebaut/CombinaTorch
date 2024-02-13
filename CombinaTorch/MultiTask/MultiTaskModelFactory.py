import torch.nn


class MultiTaskModelFactory:

    def __init__(self, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.modelclasses = {}
        self.model_parameters = {}
        self.device = device

    def get_keys(self):
        return self.modelclasses.keys()

    def add_modelclass(self, modelclass: type):
        self.modelclasses[modelclass.__name__] = modelclass

    def add_static_model_parameters(self, model_name: str, **model_parameters):
        self.model_parameters[model_name] = model_parameters

    def create_model(self, model_name: str, **dynamic_model_parameters):
        return self.modelclasses[model_name](**self.model_parameters[model_name],
                                             **dynamic_model_parameters).to(self.device)
