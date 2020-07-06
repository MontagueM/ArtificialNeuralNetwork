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
        self.grad_loss_a = None

    def set_params(self, init_b, init_W):
        self.b = np.array(init_b)
        self.W = np.array(init_W)

    def pre_activation(self):
        prev_layer = self.instance.layers[self.layer_id - 1]
        # print("W", self.W, type(self))
        # print(f'{type(prev_layer)} -> {type(self)} pre-activation')
        # print(f"Multiplying {self.W} and {prev_layer.h_x}")
        if isinstance(prev_layer, InputLayer):
            self.a_x = self.b + np.multiply(self.W, prev_layer.h_x)
        elif isinstance(self, OutputLayer):
            self.a_x = self.b + self.W.T.dot(prev_layer.h_x)
            # print("THIS1", self.W, prev_layer.h_x)
            # print("THIS2", self.W.T.dot(prev_layer.h_x))
        else:
            self.a_x = self.b + np.matmul(self.W, prev_layer.h_x)
        # print("a_x", self.a_x)
        # print("pre_activation", type(self), len(self.a_x))

    def hidden_below_grad(self):
        # print(self.W.T, self.grad_loss_a)
        grad_loss_h_below = np.matmul(self.grad_loss_a, self.W.T)
        # Sending it down to the layer below for its use in grad calculations (back propagation)
        self.instance.layers[self.layer_id - 1].grad_loss_h = grad_loss_h_below

    def hidden_below_preactivation_grad(self):
        # We're using sigmoid so need g' to be g(a)(1-g(a)) | using list comprehension as need to do element wise
        # print(self.instance.layers[self.layer_id - 1].grad_loss_h,
        #       [g(a)*(1 - g(a)) for a in self.instance.layers[self.layer_id - 1].a_x])
        grad_loss_a_below = np.multiply(self.instance.layers[self.layer_id - 1].grad_loss_h,
            [g(a)*(1 - g(a)) for a in self.instance.layers[self.layer_id - 1].a_x])
        # print(self.layer_id-1, "a1", grad_loss_a_below, '\n')
        self.instance.layers[self.layer_id - 1].grad_loss_a = grad_loss_a_below

    def W_grad(self):
        # print("W", self.grad_loss_a, np.array(self.instance.layers[self.layer_id - 1].h_x))
        if len(self.grad_loss_a) == 1 or len(self.instance.layers[self.layer_id - 1].h_x.T) == 1:
            grad_loss_W = np.multiply(self.instance.layers[self.layer_id - 1].h_x.T, self.grad_loss_a)
        else:
            grad_loss_W = np.outer(self.grad_loss_a, self.instance.layers[self.layer_id - 1].h_x)  # .outer replaces .T
        return grad_loss_W

    def b_grad(self):
        # print(self.layer_id, "a2", self.grad_loss_a)
        grad_loss_b = self.grad_loss_a
        return grad_loss_b


# k=0
class InputLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([InputUnit] * unit_count)

    def set_input_data(self, input_data):
        # self.h_x = np.array([input_data]*len(self.instance.layers[self.layer_id+1].belonging_units))
        self.h_x = np.array([input_data])
        # print(f'input h_x {self.h_x}')


# 1<=k<=L
def g(a_x: np.array):
    return 1 / (1 + np.exp(-a_x))  # Sigmoid
    # return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)  # Tanh


class HiddenLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id)
        self.belonging_units = np.array([HiddenUnit] * unit_count)
        self.grad_loss_h = None

    def activation(self):
        """
        Hidden layers have different activation functions.
        :return:
        """

        # Hidden layer activation function g(a)
        self.h_x = g(self.a_x)
        # print(f'h_x {self.h_x}')
        # print("activation", type(self), len(self.h_x))


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
        # print(f'f_x output {self.h_x}')

    def output_preactivation_grad(self, y):
        """
        Computation of the output f(x) gradient before activation: -(e(y)-f(x)). y here is the actual outcome of the
        test condition.
        :return grad_loss_a: the grad of the log likelihood with respect to a(x)
        """
        # Making the e vector
        e = [0]*(len(self.h_x) + 1)
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
        # theta of [Biases0], [Weights0], [Biases1], [Weights1], ... (biases init to 0, weights stochastically sampled)
        theta = []
        for k in range(1, len(self.layers)):  # 1 as ignoring input layer for weights
            b = np.sqrt(6) / np.sqrt(len(self.layers[k].belonging_units) + len(self.layers[k - 1].belonging_units))
            W_i = []
            # k-1 as we need to go backwards when calculating for weights
            # going i,j in this order (i being row, j column)
            for i in range(len(self.layers[k-1].belonging_units)):
                W_j = []
                for j in range(len(self.layers[k].belonging_units)):
                    # Taking H_k to be the number of neurons but really should be number of activations?
                    new_W = np.random.uniform(-b, b)
                    W_j.append(new_W)
                W_i.append(W_j)
            if len(W_i) == 1:
                W_i = W_i[0]
            theta.append(np.array([0] * len(self.layers[k].belonging_units)))
            theta.append(np.array(W_i))
        print(theta)
        return theta

    def set_params(self, theta):
        for i in range(0, len(self.layers)-1):
            self.layers[i+1].set_params(theta[i*2], theta[i*2+1])

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
            layer.pre_activation()
            layer.activation()

    # loss function; do we need to think about y?
    def loss_fn(self):
        return -np.log(self.output_layer.h_x)

    def get_all_loss_SGD(self, y):
        all_loss_SGD = []
        for i, layer in enumerate(self.layers[::-1]):  # Reversing as back prop
            if isinstance(layer, InputLayer):
                continue
            elif isinstance(layer, OutputLayer):
                layer.output_preactivation_grad(y)
            elif isinstance(layer, HiddenLayer):
                pass
            if i != len(self.layers) - 2:  # So we don't try do this stuff to input layer
                layer.hidden_below_grad()
                layer.hidden_below_preactivation_grad()
            all_loss_SGD.append(layer.b_grad())
            # print(f'adding to b {all_loss_SGD[0][-1]}')
            all_loss_SGD.append(layer.W_grad())
        # print(f'b {all_loss_SGD[0]} | W {all_loss_SGD[1]}')
        need_swap = all_loss_SGD[::-1]
        need_swap[-1] = np.array(need_swap[-1])  # a rogue list correction
        swapped = [need_swap[i:i+2][::-1] for i in range(0, len(need_swap), 2)]
        ret = np.array(swapped).flatten()
        return ret  # Reverse it back as we still need to go forward with b and W

    def iterate_SGD(self):
        """
        Iterate N times (epoch)
            For each training example x_t, y_t
                delta = -grad(loss_func(f(x_t; theta), y_t)) - lambda*grad(regulariser(theta))
                theta = theta + alpha*delta
        """
        N = 1e2
        self.alpha = 0.5
        self.lambd = 1
        for i in range(int(N)):
            for x_t, y_t in self.data:
                self.input_layer.set_input_data(x_t)
                self.set_params(self.params)
                self.forward_propagate()
                # 2 * self.theta[1] is 2*W which is grad of omega for L2
                # calling get all loss SGD requires a kept-state of a forward run for a set of h_x
                # print("SGD", -self.get_all_loss_SGD(y_t))
                # print(np.multiply(self.lambd * 2, self.params))
                loss_grad = self.get_all_loss_SGD(y_t)
                regulariser_grad = np.multiply(self.lambd * 2, np.array(self.params))
                delta = -loss_grad - regulariser_grad
                # print(f'old params {self.params}')
                self.params = self.params + self.alpha * delta
                self.params[-1] = self.params[-1][0].reshape(-1, 1)  # temp fix for weird duplication issue, fix later
                # print(f'new params {self.params}')
                print(f"WE DID IT {i}\n")


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