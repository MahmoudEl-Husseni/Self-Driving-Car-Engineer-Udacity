import numpy as np

class Node:
    def __init__(self, input_nodes:list=[]) -> None:
        
        # list on input nodes to this node
        self.input_nodes = input_nodes

        # list of nodes recieving input from this node
        self.output_nodes = []
        self.gradients = {}

        # assign this node as an output node to all its input nodes
        for in_node in self.input_nodes:
            in_node.output_nodes.append(self)
        
    def __call__(self):
        '''
        Computes the output value from this node.
        Note: every class inherit from node class need to define this method.
        '''
        raise NotImplemented
    
    def forward(self):
        '''
        Computes the output value from this node.
        Note: every class inherit from node class need to define this method.
        '''
        raise NotImplemented
    
    def backward(self):
        '''
        Computes the gradients of output of this node with respect to its input.
        '''
        raise NotImplemented

class Input(Node):
    def __init__(self):
        super().__init__()
        self.gradients = {self : 0}
        
    def __call__(self, value=None):
        self.forward(value)

    def forward(self, value=None):
        if value is not None: 
            self.value = value

    def backward(self):
        '''
        we should implement backward function in Input, as we have weights in Input form
        '''
        self.gradients = {self : 0}
        for n_out in self.output_nodes:
            self.gradients[self] += n_out.gradients[self]

class Add(Node):
    def __init__(self, *args:Node) -> None:
        super().__init__(args)

    def __call__(self):
        self.forward()

    def forward(self):
        self.value = 0
        for node in self.input_nodes:
            self.value += node.value

    def backward(self):
        
        self.gradients = {n : np.zeros_like(n.value) for n in self.input_nodes}

        # retrieve cost gradient from each of connected output
        for out_node in self.output_nodes:
            grad_cost = out_node.gradients[self]

            # update each input gradients
            for input_node in self.gradients:
                self.gradients[input_node] += np.sum(grad_cost, axis=0)
        
class Mul(Node):
    def __init__(self, *args:Node) -> None:
        super().__init__(args)

    def __call__(self):
        self.forward()

    def forward(self):
        self.value = 1
        for node in self.input_nodes:
            self.value *= node.value

    def backward(self):
        self.gradients = {n : np.zeros_like(n.value) for n in self.input_nodes}

        self.forward()
        out = self.value

        # retrieve cost gradient from each of connected output
        for out_node in self.output_nodes:
            grad_cost = out_node.gradients[self]

            # update each input gradients
            for input_node in self.input_nodes:
                self.gradients[input_node] += np.dot(grad_cost, out / input_node.value)            

class Sigmoid(Node):
    def __init__(self, input_node: Node) -> None:
        super().__init__([input_node])


    def __call__(self):
        self.forward()

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(self):
        self.value = Sigmoid._sigmoid(self.input_nodes[0].value)

    def backward(self):
        self.gradients = {n : np.zeros_like(n.value) for n in self.input_nodes}
        sig = Sigmoid._sigmoid(self.input_nodes[0].value)
        
        for out_n in self.output_nodes:
            cost_grad = out_n.gradients[self]
            self.gradients[self.input_nodes[0]] += cost_grad * sig * (1 - sig)
        
class Linear(Node):
    def __init__(self, inputs, weights, bias) -> None:
        super().__init__([inputs, weights, bias])

    def __call__(self):
        self.forward()

    def forward(self):
        out = np.dot(self.input_nodes[0].value, self.input_nodes[1].value) + self.input_nodes[2].value
        self.value = out
    
    def backward(self):
        self.gradients = {n : np.zeros_like(n.value) for n in self.input_nodes}

        for out_n in self.output_nodes:
            grad_cost = out_n.gradients[self]

            self.gradients[self.input_nodes[0]] += np.dot(grad_cost, self.input_nodes[1].value.T)      # y = X * W + b -> dy / dX = W
            self.gradients[self.input_nodes[1]] += np.dot(self.input_nodes[0].value.T, grad_cost)      # y = X * W + b -> dy / dW = X
            self.gradients[self.input_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)                           # y = X * W + b -> dy / db = 1

class MSE(Node):
    def __init__(self, y : Node, y_ : Node) -> None:
        super().__init__([y, y_])
        self.m = y.value.shape[0]
        
    def __call__(self):
        self.forward()

    def forward(self):
        y = self.input_nodes[0].value.reshape(-1, 1)
        y_ = self.input_nodes[1].value.reshape(-1, 1)
        self.diff = y - y_
        mse = np.mean(self.diff ** 2)
        self.value = mse

    def backward(self):
        self.gradients[self.input_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.input_nodes[1]] = - (2 / self.m) * self.diff

def forward_pass(output_node : Node, sorted_nodes : list):
    """
    Performs a forward pass through a list of sorted nodes.

    Args:
        output_node: A node in the graph, should be the output node (have no outgoing edges).
        sorted_nodes: A topologically sorted list of nodes.

    Returns: 
        :the output Node's value.
    """
    for n in sorted_nodes:
        n.forward()

    return output_node.value
    
def backward_prop(sorted_nodes : list):
    for n in sorted_nodes[::-1]:
        n.backward()

def sgd_update(trainables : Node, learning_rate : float=1e-3):
    """
    Updates the value of each trainable with SGD.

    Args:
        trainables: A list of `Input` Nodes representing weights/biases.
        learning_rate: The learning rate.
    """
    for n in trainables:
        n.value = n.value - learning_rate * n.gradients[n]


def topological_sort(feed_dict : dict) -> list:
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.output_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.output_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L