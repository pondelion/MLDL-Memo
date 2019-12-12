import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))


if __name__ == '__main__':

    ETA = 0.4
    ALPHA = 0.1

    input = np.arange(-4.0, 4.0, 0.1)

    output_t = []
    for i in input:
        output_t.append(0.45+0.4*math.sin(i))

    output = np.arange(-4.0, 4.0, 0.1)

    w = 0.1*np.random.random((len(input), len(input)))
    v = 0.1*np.random.random((len(input), len(input)))
    delta_w = [[0 for col in range(len(input))] for row in range(len(input))]
    delta_v = [[0 for col in range(len(input))] for row in range(len(input))]
    delta_w_old = [[0 for col in range(len(input))]
                   for row in range(len(input))]
    delta_v_old = [[0 for col in range(len(input))]
                   for row in range(len(input))]

    #print(delta_w)

    plt.plot(input, output_t)
    plt.plot(input, output)
    plt.show()

    intermediate = [0]*len(input)
    delta_output = [0]*len(output)
    delta_intermediate = [0]*len(intermediate)

    for n in range(200):

        for i in range(len(input)):
            sum = 0.0
            for j in range(len(input)):
                sum = sum + w[i, j]*input[j]
            intermediate[i] = sigmoid(sum)

        for i in range(len(input)):
            sum = 0.0
            for j in range(len(input)):
                sum = sum + v[i, j]*intermediate[j]
            output[i] = sigmoid(sum)

        for i in range(len(output)):
            delta_output[i] = output[i] * \
                (1.0 - output[i]) * (output_t[i] - output[i])

        for i in range(len(intermediate)):
            sum = 0.0
            for j in range(len(delta_output)):
                sum = sum + v[j, i] * delta_output[j]
            delta_intermediate[i] = intermediate[i] * \
                (1.0 - intermediate[i]) * sum

        for k in range(len(output)):
            for j in range(len(output)):
                delta_v[k][j] = ETA * delta_output[k] * \
                    intermediate[j] + ALPHA * delta_v_old[k][j]
                v[k, j] = v[k, j] + delta_v[k][j]
                delta_v_old[k][j] = delta_v[k][j]

        for j in range(len(input)):
            for i in range(len(input)):
                delta_w[j][i] = ETA * delta_intermediate[j] * \
                    input[i] + ALPHA * delta_w_old[j][i]
                w[j, i] = w[j, i] + delta_w[j][i]
                delta_w_old[j][i] = delta_w[j][i]

    plt.plot(input, output_t)
    plt.plot(input, output)
    plt.show()
