import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data=pd.read_csv("trainRegression.csv")
x=np.array(train_data['X'])
y=np.array(train_data['R'])

A=np.empty((4,4),dtype=float)
for i in range(4):
    for j in range(4):
        A[i,j]=np.sum(x**(i+j)) if (i+j)!=0 else len(x)
#print("Matrix A: ",A)

B=np.empty((4,1),dtype=float)
for i in range(4):
    B[i, 0] = np.sum(y * x**i)
#print("Matrix B: ",B)

ans=np.linalg.solve(A,B)
print("Constants: ",ans)

predtrain_data=ans[0,0]+(ans[1,0]*x)+(ans[2,0]*x**2)+(ans[3,0]*x**3)
print("Prediction on train_data: ",predtrain_data)

plt.plot(x,y,'.')
plt.title("Training Data")
plt.xlabel("X-values")
plt.ylabel("R-values")
plt.plot(x,predtrain_data)
plt.show()

MSE_traindata=np.sum((predtrain_data-y)*(predtrain_data-y))/len(x)
print("Mean Square Error(Train Data): ",MSE_traindata)

test_data=pd.read_csv("testRegression.csv")
x1=np.array(test_data['X'])
y1=np.array(test_data['R'])

predtest_data=ans[0,0]+(ans[1,0]*x1)+(ans[2,0]*x1**2)+(ans[3,0]*x1**3)
print("Prediction on test_data: ",predtest_data)

plt.title("Testing Data")
plt.plot(x1,y1,'.')
plt.xlabel("X-values")
plt.ylabel("R-values")
plt.plot(x1,predtest_data)
plt.show()

MSE_testdata=np.sum((predtest_data-y1)*(predtest_data-y1))/len(x1)
print("Mean Square Error(test data): ",MSE_testdata)

#Result: The cubic regression model captures the training data closely but shows a higher error on test data, indicating possible overfitting.