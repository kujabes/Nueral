import numpy as np

softmax_output = np.array([0.7, 0.1, 0.2])
target_output = np.array([1, 0, 0])

loss = -(target_output * np.log(softmax_output))
print(sum(loss))

loss2 = np.power(softmax_output - target_output, 2)
print(sum(loss2))