import numpy as np

# x: Features tensor (samples and features), y: target tensor, learning_rate: step size, num_iterations: number of iterations
def gradient_descent(x, y, learning_rate, num_iterations):
    samples = x.shape[0]
    features = x.shape[1]

    theta = np.zeros(features)
    for i in range(num_iterations):
        gradient = np.dot(x.T, np.dot(x, theta) - y) / samples
        theta = theta - learning_rate * gradient
    return theta

def main():
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([3, 4, 5, 6, 7])
    # maxes around 0.1
    learning_rate = 0.1
    num_iterations = 1000

    theta = gradient_descent(x, y, learning_rate, num_iterations)
    print(theta)

if __name__ == '__main__':
    main()