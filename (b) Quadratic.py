import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data=pd.read_csv("trainRegression.csv")
x=np.array(train_data['X'])
y=np.array(train_data['R'])

A=np.empty((3,3),dtype=float)
for i in range(3):
    for j in range(3):
        A[i,j]=np.sum(x**(i+j)) if (i+j)!=0 else len(x)
#print("Matrix A: ",A)

B=np.empty((3,1),dtype=float)
for i in range(3):
    B[i,0]=np.sum(y*x**i)
#print("Matrix B: ",B)

ans=np.linalg.solve(A,B)
print("Constants: ",ans)

predtrain_data=ans[0,0]+(ans[1,0]*x)+(ans[2,0]*x**2)
print("Prediction on Training Data: ",predtrain_data)
#print(np.shape(predtrain_data))

plt.plot(x,y,'.')
plt.title("Training data")
plt.xlabel("X-values")
plt.ylabel("R-values")
plt.plot(x,predtrain_data)
plt.show()

MSE_traindata=(np.sum((predtrain_data-y)*(predtrain_data-y)))/len(x)
print("Mean Square Error(train_data): ",MSE_traindata)

test_data=pd.read_csv("testRegression.csv")
#print(test_data.head(n=8))
x1=np.array(test_data['X'])
y1=np.array(test_data['R'])

predtest_data=ans[0,0]+(ans[1,0]*x1)+(ans[2,0]*x1**2)
print("Prediction on test_data: ",predtest_data)

plt.plot(x1,y1,'.')
plt.plot(x1,predtest_data)
plt.title("Test data")
plt.xlabel("X-values")
plt.ylabel("R-values")
plt.show()

MSE_testdata=np.sum(ans[0,0]+(ans[1,0]*x1)+(ans[2,0]*x1**2))
print("Mean Square error(test_data): ",MSE_testdata)

#Result: The quadratic regression model fits the training data better than a linear model,but its test error suggests it may not generalize well.