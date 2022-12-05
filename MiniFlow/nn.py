from Node import *
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Configurations
#---------------
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
OUTPUT_SHAPE = 1
HIDDEN_LAYER1 = 10

np.random.seed(42)
# ================================ #

# prepare data
_X, _y = load_boston(return_X_y=True)
data = np.concatenate([_X, _y.reshape(-1, 1)], axis=1)

X = StandardScaler().fit_transform(X=_X)
m, input_shape = X.shape

# Network Architecture: Input -> Linear(input_shape -> 10) -> sigmoid -> Output(10 -> 1)
X, y = Input(), Input()
X.forward(_X)
y.forward(_y)

w1, b1 = Input(), Input()
w2, b2 = Input(), Input()

L1 = Linear(X, w1, b1)
Act = Sigmoid(L1)

# Define loss function
out = Linear(Act, w2, b2)
Loss = MSE(y, out)

# initialize weights
w1.forward(np.random.randn(input_shape, HIDDEN_LAYER1))
b1.forward(np.random.rand(10))
w2.forward(np.random.randn(HIDDEN_LAYER1, OUTPUT_SHAPE))
b2.forward(np.random.rand(OUTPUT_SHAPE))

trainables = [w1, b1, w2, b2]
graph = [
    X, w1, b1,  
    L1,
    Act, w2, b2,  
    out,
    y, Loss
]


# train model
steps_per_epoch = m // BATCH_SIZE
history = []
for epoch in range(EPOCHS):
    loss = []
    np.random.shuffle(data)
    for step in range(steps_per_epoch):
        x_ = data[step*BATCH_SIZE : (step+1) * BATCH_SIZE, :-1]
        y_ = data[step*BATCH_SIZE : (step+1) * BATCH_SIZE, -1]

        X.forward(x_)
        y.forward(y_)

        forward_pass(Loss, graph)

        backward_prop(graph)
        sgd_update(trainables=trainables, learning_rate=LR)
        
        _loss = graph[-1].value
        loss.append(_loss)

    history.append(np.mean(loss))
    print(f"{f'[INFO]: Epoch {epoch + 1}'.ljust(25, '.')}->  MSE: {np.mean(loss) / steps_per_epoch}")

# import plotly.graph_objects as go

# fig = go.Figure(go.Scatter(x=np.arange(EPOCHS), y=history))
# fig.show()