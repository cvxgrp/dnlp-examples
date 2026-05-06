import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(0)


# --------------------------------------------------------------------
#    generate synthetic data and specify neural network architecture
# --------------------------------------------------------------------

# generate synthetic 2D dataset (moons) for binary classification
N = 200
X, y_01 = make_moons(n_samples=N, noise=0.30, random_state=1)
y = 2 * y_01 - 1  # convert from {0,1} to {-1,1}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
N_train = X_train.shape[0]

# change dimensions for neat cvxpy code (each sample is a column)
y_train = y_train.reshape(1, -1)  # (1, N_train)
X_train = X_train.T # (n0, N_train)  

# define dimensions of neural network and regularization strength
n0, n1, L = 2, 4, 2
lmbda = 0.001


# ----------------------------------------------------------------
#     the actual optimization problem for training the network
# ---------------------------------------------------------------
# define weights and biases of network
W1 = cp.Variable((n1, n0))  
v1 = cp.Variable((n1, 1))   
W2 = cp.Variable((1, n1))   
v2 = cp.Variable()       

Z1 = cp.tanh(W1 @ X_train + v1)
psi = W2 @ Z1 + v2
logistic_loss = cp.sum(cp.logistic(-cp.multiply(y_train, psi))) / N_train
regularization = lmbda * (cp.sum_squares(W1) + cp.sum_squares(W2))
objective = cp.Minimize(logistic_loss + regularization)
prob = cp.Problem(objective)

# intialize weights and solve
W1.value = np.random.randn(n1, n0)
prob.solve(nlp=True, solver=cp.IPOPT, verbose=True)


#----------------------------------------------------------------
#               rest of the code is visualization
# ---------------------------------------------------------------

# evaluate on training set
Z1_train = np.tanh(W1.value @ X_train + v1.value)
psi_train = (W2.value @ Z1_train)[0, :] + v2.value
y_pred_train = np.sign(psi_train)
acc_train = accuracy_score(y_train.ravel(), y_pred_train)
print(f"Train accuracy: {acc_train:.2%}")

# evaluate on test set
Z1_test = np.tanh(W1.value @ X_test.T + v1.value)
psi_test = (W2.value @ Z1_test)[0, :] + v2.value
y_pred = np.sign(psi_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2%}")

# plot decision boundary with training points
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]

Z1_grid = np.tanh(W1.value @ grid.T + v1.value)
psi_grid = (W2.value @ Z1_grid)[0, :] + v2.value
zz = psi_grid.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.contourf(xx, yy, zz, levels=[-1e6, 0, 1e6], colors=["#AEDFF7", "#FFD6D6"],
            alpha=0.6)
ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.0)

colors = {-1: "tab:blue", 1: "tab:red"}
for label in [-1, 1]:
    mask = y_train.ravel() == label
    ax.scatter(X_train[0, mask], X_train[1, mask], c=colors[label],
               edgecolors="k", s=30, linewidths=0.5, label=f"y = {label}")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
fig.tight_layout()
fig.savefig(f"figures/neural_net_classification_lmbda={lmbda}.pdf")
