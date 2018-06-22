import tensorflow as tf
import numpy as np


a = [1,2,3,4,5]
c = [4,4,6,5,6]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

m = tf.Variable(3,dtype = np.float32,name = 'Slope')
b = tf.Variable(-3,dtype = np.float32,name = 'Intercept')

#Line equation 

y_predict = tf.multiply(m,x) + b

squarederror = tf.square(y_predict-y)

cost = tf.reduce_sum(squarederror)

optimizing = tf.train.GradientDescentOptimizer(0.0001)

train = optimizing.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	sess.run(train,feed_dict = {x:a,y:c})
	M = sess.run(m)
	B = sess.run(b)
print(M)
print(B)

PredictX = 4
Y_optimizedPredict = M * PredictX + B

print(Y_optimizedPredict)
