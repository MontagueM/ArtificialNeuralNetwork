import numpy as np

class Unit:
    pass


class InputUnit(Unit):
    pass


class HiddenUnit(Unit):
    pass


class OutputUnit(Unit):
    pass


class Layer:
    def __init__(self, instance, layer_id, unit_count=1):
        self.instance = instance
        self.layer_id = layer_id
        # We'll need to change this design when we initialise hyperparams for this? or do we? idk
        self.belonging_units = np.array([Unit]*unit_count)

        self.a_x = None
        self.h_x = None
        self.b = None
        self.W = None

    def init_params(self, init_b, init_W):
        self.b = init_b
        self.W = init_W

    # Back propagation fns needed here for updating b and W

    def pre_activation(self):
        prev_layer = self.instance.layers[self.layer_id - 1]
        self.a_x = self.b + self.W.dot(prev_layer.h_x)

    def activation(self):
        pass


# k=0
class InputLayer(Layer):
    def __init__(self, instance, layer_id, input_data, unit_count=1):
        super().__init__(instance, layer_id)
        self.h_x = input_data
        self.belonging_units = np.array([InputUnit] * unit_count)


# 1<=k<=L
class HiddenLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([HiddenUnit] * unit_count)

    def activation(self):
        """
        Hidden layers have different activation functions.
        :return:
        """
        def g(a_x: np.array):
            return 1 / (1 + np.exp(-a_x))  # Sigmoid
            # return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)  # Tanh

        # Hidden layer activation function g(a)
        self.h_x = g(self.a_x)



# k=L+1
class OutputLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([HiddenUnit] * unit_count)

    def activation(self):
        """
        Output layers have different activation functions.
        :return:
        """
        a_x_sum = sum(self.a_x)
        # Output activation function o(a)
        o = np.array([np.exp(i)/a_x_sum for i in self.a_x]).T
        self.h_x = o  # also f_x

    def output_grad(self, y):
        """
        Computation of the output f(x) gradient before activation: -(e(y)-f(x)). y here is the actual outcome of the
        test condition.
        :return grad_a_LL: the grad of the log likelihood with respect to a(x)
        """
        # Making the e vector
        e = [0]*len(self.h_x)
        e[y] = 1

        grad_LL_a = ([-e[i] + self.h_x[i] for i in range(len(self.h_x))])


class NNInstance:
    """
    An instance encapsulating all layers and units.
    """
    def __init__(self, f_input):
        def flatten(something):
            if isinstance(something, list):
                for sub in something:
                    yield from flatten(sub)
            else:
                yield something
        # Initialising units and layers
        input_layer = InputLayer(self, 0, f_input, unit_count=1)
        hidden_layers = [HiddenLayer(self, 0, unit_count=3), HiddenLayer(self, 1, unit_count=3)]
        output_layer = OutputLayer(self, 0, unit_count=1)
        self.layers = list(flatten([input_layer, hidden_layers, output_layer]))
        print(self.layers)
        theta = self.get_initialisation_params()
        self.set_init_params(theta)
    # Loss function
    # Gradient computation procedure

    # Regulariser
    def regulariser(self):
        pass

    # Initialisation method
    def get_initialisation_params(self):
        # theta of [Biases], [Weights] (biases init to 0, weights stochastically sampled)
        theta = [[0]*(len(self.layers) - 1), []]
        for i in range(1, len(self.layers)):  # 1 as ignoring input layer for weights
            b = np.sqrt(6) / np.sqrt(len(self.layers[i].belonging_units) + len(self.layers[i - 1].belonging_units))

            W = []
            for j in range(len(self.layers[i].belonging_units)):
                # Taking H_k to be the number of neurons but really should be number of activations?
                new_W = np.random.uniform(-b, b)
                W.append(new_W)
            theta[1].append(W)
        print(theta)
        return theta

    def set_init_params(self, theta):
        for i in range(0, len(self.layers)-1):
            self.layers[i].init_params(theta[0][i], theta[1][i])


if __name__ == "__main__":
    # Some dummy data that links to 0 if <0.5 and 1 if >0.5
    input_data = np.linspace(0, 1, 1000)
    output_data = [0]*500 + [1]*500
    init_input_data = input_data[0]
    inst = NNInstance(init_input_data)
