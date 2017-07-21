# Python imports
import numpy  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library

# Allow matplotlib to plot inside this notebook matplotlib inline
# Set the seed of the numpy random number generator so that the tutorial is reproducable
numpy.random.seed(seed=1)

# Define the vector of input samples as x, with 20 values sampled from a uniform distribution
# between 0 and 1
x = numpy.random.uniform(0, 1, 20)


# Generate the target values t from x with small gaussian noise so the estimation won't
# be perfect.
# Define a function f that represents the line that generates t without noise
def f(x): return x * 2


# Create the targets t with some gaussian noise
noise_variance = 0.2  # Variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = numpy.random.randn(x.shape[0]) * noise_variance
# Create targets t
t = f(x) + noise

# Plot the target t versus the input x
"""
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$t$', fontsize=15)
plt.ylim([0, 2])
plt.title('inputs (x) vs targets (t)')
plt.grid()
plt.legend(loc=2)
plt.show()
"""

def nn(x, w): return x * w


# Define the cost function
def cost(y, t): return ((t - y) ** 2).sum()


# Plot the cost vs the given weight w

# Define a vector of weights for which we want to plot the cost
ws = numpy.linspace(0, 4, num=100)  # weight values
cost_ws = numpy.vectorize(lambda w: cost(nn(x, w), t))(ws)  # cost for each weight in ws

""""
# Plot
plt.plot(ws, cost_ws, 'r-')
plt.xlabel('$w$', fontsize=15)
plt.ylabel('$xi$', fontsize=15)
plt.title('cost vs. weight')
plt.grid()
plt.show()
"""

# define the gradient function. Remember that y = nn(x, w) = x * w
def gradient(w, x, t):
    return 2 * x * (nn(x, w) - t)


# define the update function delta w
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()


# Set the initial weight parameter
w = 0.1
# Set the learning rate
learning_rate = 0.1

# Start performing the gradient descent updates, and print the weights and cost:
nb_of_iterations = 5  # number of gradient descent updates
w_cost = [(w, cost(nn(x, w), t))]  # List to store the weight,costs values
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)  # Get the delta w update
    w = w - dw  # Update the current weight parameter
    w_cost.append((w, cost(nn(x, w), t)))  # Add weight,cost to list

# Print the final w, and cost
for i in range(0, len(w_cost)):
    print('w({}): {:.4f} t cost: {:.4f}'.format(i, w_cost[i][0], w_cost[i][1]))


# define the gradient function. Remember that y = nn(x, w) = x * w
def gradient(w, x, t):
    return 2 * x * (nn(x, w) - t)


# define the update function delta w
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()


# Plot the first 2 gradient descent updates
plt.plot(ws, cost_ws, 'r-')  # Plot the error curve
# Plot the updates
for i in range(1, len(w_cost) - 2):
    w1, c1 = w_cost[i - 1]
    w2, c2 = w_cost[i]
    plt.plot(w1, c1, 'bo')
    plt.plot([w1, w2], [c1, c2], 'b-')
    plt.text(w1, c1 + 0.5, '$w({})$'.format(i))
# Plot the last weight, axis, and show figure
w1, c1 = w_cost[len(w_cost) - 3]
plt.plot(w1, c1, 'bo')
plt.text(w1, c1 + 0.5, '$w({})$'.format(nb_of_iterations))
plt.xlabel('$w$', fontsize=15)
plt.ylabel('$xi$', fontsize=15)
plt.title('Gradient descent updates plotted on cost function')
plt.grid()
plt.show()

w = 0
# Start performing the gradient descent updates
nb_of_iterations = 10  # number of gradient descent updates
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)  # get the delta w update
    w = w - dw  # update the current weight parameter

# Plot the fitted line agains the target line
# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0 * w, 1 * w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0, 2])
plt.title('input vs. target')
plt.grid()
plt.legend(loc=2)
plt.show()


