from ._defaults import _load_defaults

NN = None
NN2 = None
NN_NAME = None
PREVIOUS_STAGE = None


def load_nn(stage: int = 3):
    global NN
    global NN_NAME
    global PREVIOUS_STAGE

    if NN_NAME != _load_defaults()['Demo']['NeuralNetworkPath'] or stage != PREVIOUS_STAGE:
        # tf may be absent sometimes or may load for some time
        # import it only if needed
        from tensorflow.keras.models import load_model
        from sweet.optimization.nn.nn import NNFactory
        from pathlib import Path

        NN_NAME = _load_defaults()['Demo']['NeuralNetworkPath']
        NN_PATH = (Path(__file__).parents[3] / 'data' / NN_NAME).resolve()
        NN = load_model(NN_PATH, compile=False)
        if stage==3:
            return NN
        
        import tensorflow as tf
        NN = tf.keras.models.Model(NN.input, [NN.layers[-4].output, NN.layers[-5].output])
    return NN


def load_new_nn(nn_name=None):
    global NN2
    assert nn_name is not None, "You don't put name of NN"

    # tf may be absent sometimes or may load for some time
    # import it only if needed
    from tensorflow.keras.models import load_model
    from sweet.optimization.nn.nn import NNFactory
    from pathlib import Path

    NN_NAME = _load_defaults()['Demo']['NeuralNetworkPath']
    NN_PATH = (Path(__file__).parents[3] / 'data' / nn_name).resolve()
    NN2 = load_model(NN_PATH, compile=False)
    return NN2
