import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([4,4,6,5,6])

m = -14.9
b = 0
learningRate = 0.0001
epoch = 1000
N = len(y)

for i in range(epoch):
	y_current = x*m +b
	m_gradient = (-2/N)*sum(x*(y-y_current))
	b_gradient = (-2/N)*sum(y-y_current)
	m = m - (learningRate*m_gradient)
	b = b - (learningRate*b_gradient)

print(m)
print(b)

y_predict = m*4+b

print(y_predict)