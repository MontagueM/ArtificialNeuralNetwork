import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, instance, layer_id, unit_count=1):
        self.instance = instance
        self.layer_id = layer_id
        # We'll need to change this design when we initialise hyperparams for this? or do we? idk
        self.unit_count = unit_count

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
        if isinstance(prev_layer, InputLayer):
            self.a_x = self.b + np.matmul(prev_layer.h_x, self.W)
        elif isinstance(self, OutputLayer):
            self.a_x = self.b + np.matmul(prev_layer.h_x, self.W)
        else:
            self.a_x = self.b + np.matmul(self.W, prev_layer.h_x)

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
            grad_loss_W = np.outer(self.instance.layers[self.layer_id - 1].h_x, self.grad_loss_a)  # .outer replaces .T

        return grad_loss_W

    def b_grad(self):
        # print(self.layer_id, "a2", self.grad_loss_a)
        grad_loss_b = self.grad_loss_a
        return grad_loss_b


# k=0
class InputLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id, unit_count)

    def set_input_data(self, input_data):
        self.h_x = np.array(input_data)


# 1<=k<=L
def g(a_x: np.array):
    return 1 / (1 + np.exp(-a_x))  # Sigmoid
    # return (np.exp(2*a) - 1)/(np.exp(2*a) + 1)  # Tanh


class HiddenLayer(Layer):
    def __init__(self, instance, layer_id, unit_count=1):
        super().__init__(instance, layer_id, unit_count)
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
        super().__init__(instance, layer_id, unit_count)
        self.grad_loss_a = None

    def activation(self):
        """
        Output layers have different activation functions.
        :return:
        """
        # print("a_x", self.a_x)
        a_x_sum = sum(self.a_x)
        # Output activation function o(a), softmax
        # o = np.array([np.exp(i)/a_x_sum for i in self.a_x]).T
        # Sigmoid activation
        o = np.array([1/(1+np.exp(-i)) for i in self.a_x]).T
        self.h_x = o  # also f_x
        # print(f'f_x output {self.h_x}')

    def output_preactivation_grad(self, y):
        """
        Computation of the output f(x) gradient before activation: -(e(y)-f(x)). y here is the actual outcome of the
        test condition.
        :return grad_loss_a: the grad of the log likelihood with respect to a(x)
        """
        # Making the e vector
        e = [0]*(len(self.h_x))  # + 1)
        if len(e) == 1:
            if y == 1:
                e = [1]
        else:
            e[y] = 1
        self.grad_loss_a = ([-e[i] + self.h_x[i] for i in range(len(self.h_x))])


class NNInstance:
    """
    An instance encapsulating all layers and units.
    """
    def __init__(self, training_data, validation_data, test_data):
        def flatten(something):
            if isinstance(something, list):
                for sub in something:
                    yield from flatten(sub)
            else:
                yield something
        # Initialising units and layers
        self.input_layer = InputLayer(self, 0, unit_count=2)
        hidden_layers = [HiddenLayer(self, 1, unit_count=3), HiddenLayer(self, 2, unit_count=3)]
        self.output_layer = OutputLayer(self, 3, unit_count=2)
        self.layers = list(flatten([self.input_layer, hidden_layers, self.output_layer]))
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.params = self.get_initialisation_params()
        # Run forward once to get stuff we need for back prop

    # Initialisation method
    def get_initialisation_params(self):
        # theta of [Biases0], [Weights0], [Biases1], [Weights1], ... (biases init to 0, weights stochastically sampled)
        theta = []
        for k in range(1, len(self.layers)):  # 1 as ignoring input layer for weights
            b = np.sqrt(6) / np.sqrt(self.layers[k].unit_count + self.layers[k - 1].unit_count)
            W_i = []
            # k-1 as we need to go backwards when calculating for weights
            # going i,j in this order (i being row, j column)
            for i in range(self.layers[k-1].unit_count):
                W_j = []
                for j in range(self.layers[k].unit_count):
                    # Taking H_k to be the number of neurons but really should be number of activations?
                    new_W = np.random.uniform(-b, b)
                    W_j.append(new_W)
                W_i.append(W_j)
            if len(W_i) == 1:
                W_i = W_i[0]
            theta.append(np.array([0] * self.layers[k].unit_count))
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
        W = [self.params[i+1] for i in range(0, len(self.params), 2)]
        for layer_W in W:
            omega += np.linalg.norm(layer_W)**2
        return omega

    def forward_propagate(self):
        for layer in self.layers[1:]:  # 1 as ignoring input layer
            layer.pre_activation()
            layer.activation()
        # print(self.layers[-1], self.layers[-1].h_x)
        return self.layers[-1].h_x  # f_x result

    # loss function; do we need to think about y?
    def loss_fn(self, f_x):
        return -np.log(f_x)

    def get_generalisation_error(self, loss_test_array):
        return (1/len(self.test_data)) * sum(loss_test_array)# + self.lambd * self.l2_regularisation() TODO fix

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
            all_loss_SGD.append(layer.W_grad())

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
        N = 1e1
        self.alpha = 0.5
        self.lambd = 1
        for i in range(int(N)):
            for x_t, y_t in self.training_data:
                self.input_layer.set_input_data(x_t)
                self.set_params(self.params)
                self.forward_propagate()
                # calling get all loss SGD requires a kept-state of a forward run for a set of h_x
                loss_grad = self.get_all_loss_SGD(y_t)
                regulariser_grad = np.multiply(self.lambd * 2, np.array([[0, self.params[i]] for i in range(0, len(self.params), 2)]).flatten())
                delta = -loss_grad #- regulariser_grad  # TODO remove this # as debug mode
                self.params = self.params + self.alpha * delta
                print(f"WE DID IT {i}\n")

            test_accuracy = self.test_set(self.test_data)
            test_accuracies.append(test_accuracy)

    def test_set(self, test_data):
        test_results = []
        loss_test_array = []
        for x_t, y_t in test_data:
            self.input_layer.set_input_data(x_t)
            f_x = self.forward_propagate()
            loss_test_array.append(self.loss_fn(f_x[y_t]))
            if f_x[y_t] > 0.5:
                test_results.append(True)
            else:
                test_results.append(False)
        print(f'Gen error: {self.get_generalisation_error(loss_test_array)}')
        gen_errors.append(self.get_generalisation_error(loss_test_array))
        return test_results


if __name__ == "__main__":
    test_accuracies = []
    gen_errors = []
    # Some dummy data that links to 0 if <0.5 and 1 if >0.5
    data = []
    for i in range(1000):
        rand = np.random.randint(0, 2)
        if rand == 0:
            inp = [(np.random.random(1)[0], np.random.random(1)[0]*0.5), 0]
        else:
            inp = [(1-(np.random.random(1)[0]), 1-(np.random.random(1)[0]*0.5)), 1]
        data.append(inp)

    training_data = data[:700]
    validation_data = data[700:850]
    test_data = data[850:]

    plt.plot([x[0] for x, y in data if y == 0], [x[1] for x, y in data if y == 0], 'o')
    plt.plot([x[0] for x, y in data if y == 1], [x[1] for x, y in data if y == 1], 'x')
    plt.show()
    inst = NNInstance(training_data, validation_data, test_data)
    inst.iterate_SGD()

    [print(f'{round(x.count(True)*100/len(x), 1)}% | {x.count(True)}/{len(x)} Correct') for x in test_accuracies]
    print(f'Gen error: {gen_errors}')
