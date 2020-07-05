import numpy as np
import matplotlib.pyplot as plt

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

    def set_params(self, init_b, init_W):
        self.b = np.array(init_b)
        self.W = np.array(init_W)

    def pre_activation(self):
        prev_layer = self.instance.layers[self.layer_id - 1]
        print(prev_layer.h_x, type(self), type(prev_layer))
        self.a_x = self.b + np.matmul(self.W.T, prev_layer.h_x)


# k=0
class InputLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([InputUnit] * unit_count)

    def set_input_data(self, input_data):
        self.h_x = input_data

# 1<=k<=L
def g(a_x: np.array):
    return 1 / (1 + np.exp(-a_x))  # Sigmoid
    # return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)  # Tanh


class HiddenLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([HiddenUnit] * unit_count)
        self.grad_loss_a = None
        self.grad_loss_h = None
        self.grad_loss_W = None
        self.grad_loss_b = None

    def activation(self):
        """
        Hidden layers have different activation functions.
        :return:
        """

        # Hidden layer activation function g(a)
        self.h_x = g(self.a_x)

    def hidden_below_grad(self):
        grad_loss_h_below = np.matmul(self.W.T, self.grad_loss_a)
        # Sending it down to the layer below for its use in grad calculations (back propagation)
        self.instance.layers[self.layer_id - 1].grad_loss_h = grad_loss_h_below

    def hidden_below_preactivation_grad(self):
        # We're using sigmoid so need g' to be g(a)(1-g(a)) | using list comprehension as need to do element wise
        grad_loss_a_below = self.instance.layers[self.layer_id - 1].grad_loss_h.dot([g(a)(1 - g(a)) for a in self.layer_below.a_x])
        self.instance.layers[self.layer_id - 1].grad_loss_a = grad_loss_a_below

    def hidden_W_grad(self):
        self.grad_loss_W = np.matmul(self.grad_loss_a, self.instance.layers[self.layer_id - 1].h_x.T)

    def hidden_b_grad(self):
        self.grad_loss_b = self.grad_loss_a


# k=L+1
class OutputLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([HiddenUnit] * unit_count)
        self.grad_loss_a = None

    def activation(self):
        """
        Output layers have different activation functions.
        :return:
        """
        a_x_sum = sum(self.a_x)
        # Output activation function o(a)
        o = np.array([np.exp(i)/a_x_sum for i in self.a_x]).T
        self.h_x = o  # also f_x

    def output_preactivation_grad(self, y):
        """
        Computation of the output f(x) gradient before activation: -(e(y)-f(x)). y here is the actual outcome of the
        test condition.
        :return grad_loss_a: the grad of the log likelihood with respect to a(x)
        """
        # Making the e vector
        e = [0]*len(self.h_x)
        e[y] = 1

        self.grad_loss_a = ([-e[i] + self.h_x[i] for i in range(len(self.h_x))])


class NNInstance:
    """
    An instance encapsulating all layers and units.
    """
    def __init__(self, data):
        def flatten(something):
            if isinstance(something, list):
                for sub in something:
                    yield from flatten(sub)
            else:
                yield something
        # Initialising units and layers
        self.input_layer = InputLayer(self, 0, unit_count=1)
        hidden_layers = [HiddenLayer(self, 1, unit_count=3), HiddenLayer(self, 2, unit_count=3)]
        self.output_layer = OutputLayer(self, 3, unit_count=1)
        self.layers = list(flatten([self.input_layer, hidden_layers, self.output_layer]))
        print(self.layers)
        self.data = data
        self.params = self.get_initialisation_params()
        # self.set_params(self.params)
        # Run forward once to get stuff we need for back prop

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

    def set_params(self, theta):
        for i in range(0, len(self.layers)-1):
            self.layers[i].set_params(theta[0][i], theta[1][i])

    # Initialise hyperparameters

    # Regularisation (using L2 for now)
    def l2_regularisation(self):
        # Grad omega is just 2W
        omega = 0
        for layer in self.layers:
            omega += np.linalg.norm(layer.W, ord='fro')**2
        return omega

    def forward_propagate(self):
        for layer in self.layers[1:]:  # 1 as ignoring input layer
            for j in range(len(layer.belonging_units)):
                layer.pre_activation()
                layer.activation()

    # loss function; do we need to think about y?
    def loss_fn(self):
        return -np.log(self.output_layer.h_x)

    def get_all_loss_SGD(self):
        all_loss_SGD = [[], []]
        for layer in self.layers[::-1]:  # Reversing as back prop
            if isinstance(layer, HiddenLayer):
                all_loss_SGD[0].append(layer.hidden_b_grad())
                all_loss_SGD[1].append(layer.hidden_W_grad())

        return np.array([x[::-1] for x in all_loss_SGD])  # Reverse it back as we still need to go forward with b and W

    def iterate_SGD(self):
        """
        Iterate N times (epoch)
            For each training example x_t, y_t
                delta = -grad(loss_func(f(x_t; theta), y_t)) - lambda*grad(regulariser(theta))
                theta = theta + alpha*delta
        """
        N = 1e3
        self.alpha = 0.5
        self.lambd = 1
        for i in range(int(N)):
            for x_t, y_t in self.data:
                self.input_layer.set_input_data(x_t)
                self.set_params(self.params)
                self.forward_propagate()
                # 2 * self.theta[1] is 2*W which is grad of omega for L2
                # calling get all loss SGD requires a kept-state of a forward run for a set of h_x
                delta = -self.get_all_loss_SGD() - self.lambd * 2*self.params[1]
                self.params = self.params + self.alpha * delta


if __name__ == "__main__":
    # Some dummy data that links to 0 if <0.5 and 1 if >0.5
    input_data = np.linspace(0, 1, 1000)
    output_data = [0]*500 + [1]*500
    data = [(input_data[i], output_data[i]) for i in range(len(input_data))]
    np.random.shuffle(data)
    actual_data = data[:50]
    plt.scatter([x[0] for x in actual_data], [x[1] for x in actual_data])
    plt.show()
    inst = NNInstance(actual_data)
    inst.iterate_SGD()