'''
Written by : Keerthi Kumar K.G

Simple Linear Regression using python from scratch 
using the equation:
where:
	m is the slope of the Line
	b is the y intercept of the Line

y = m*x + b

so if w have to find y then firstly we have to find m and then b

for finding m we can use the statistics formulae that is:
Where:
	x_:is the mean of x values
	y_:is the mean of y values
	xy_:is the mean of x*y values
	(x**2)_:is the mean of x square values

	m = ((x_*y_)-(xy_))/(((x_)**2)-(x**2)_)
	b = y_ - m*x_

and then we find the best fit line by using R-Squared formulae
That's theory required for writting the code logic

Now here we have taken the salary datasets based on the experience 
'''
import operator
import numpy as np
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt     
from sklearn import cross_validation

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values

list = x.tolist()
list2 = reduce(operator.add,list)

X = np.array(list2)

Y = dataset.iloc[:,1].values

#print(X)
#print(Y)

# regressionLine = m*x + b

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)

def slopeOfLine(x,y):
	a = ((mean(x)*mean(y)) - (mean(x*y)))
	b = ((mean(x)**2) - (mean(x**2)))
	global m
	m = a/b
	return m

def interceptOfLine(x,y):
	b = mean(y) - (m*mean(x))
	return b
slope = slopeOfLine(X_train,Y_train)
intercept = interceptOfLine(X_train,Y_train)

#print(slope)
#print(intercept)

def squaredError(y_original,y_regression):
	return sum((y_regression-y_original)**2)

def coefficientOfDetermination(y_original,y_regression):
	y_mean_line = [mean(y_original) for y in y_original]
	Squareerror = squaredError(y_original,y_regression)
	Squaremeanerror = squaredError(y_original,y_mean_line)
	return 1 - (Squareerror/Squaremeanerror)

def predict(X_Predict):
	Y_Predict = slope*X_Predict + intercept
	print 'Salary for the', X_Predict , 'given year of experience is', Y_Predict
	plt.scatter(X_Predict,Y_Predict,s = 50,color = 'yellow',label = 'TestingData')
	plt.scatter(X_train,Y_train,s = 50 ,color = 'red',label = 'TrainingData')
	plt.plot(X_train,regressionLine,color = 'green',label = 'RegressionBestFitLine')
	plt.xlabel('Salary in dollars')
	plt.ylabel('Experience in years')
	plt.title('Applying Linear Regression against the data Salary versus Experience')
	plt.legend()
	plt.show()

	return Y_Predict

regressionLine = []

for xs in X_train:
	regression = slope*xs + intercept
	regressionLine.append(regression)

rSquared = coefficientOfDetermination(Y_train,regressionLine)

print'The coefficient Of determination value is ', rSquared

Predict = predict(1.1)