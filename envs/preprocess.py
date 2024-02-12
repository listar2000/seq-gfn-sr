from gfn.preprocessors import Preprocessor
from gfn.states import States
from torchtyping import TensorType as TT


class OneHotLSTMPreprocessor(Preprocessor):
    def preprocess(self, states: States) -> TT["batch_shape", "input_dim"]:
        pass
