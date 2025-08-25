import numpy as np
import pandas as pd

train_X=np.genfromtxt("trainX.txt",delimiter=' ',dtype=int)
train_Y=pd.read_csv("trainY.txt",header=None).values.flatten()
test_X=np.genfromtxt("testX.txt",delimiter=' ',dtype=int)
test_Y=pd.read_csv("testY.txt",header=None).values.flatten()

two_train=np.array(train_X[:250,:])
four_train=np.array(train_X[250:500,:])

prob_two_1=(two_train.sum(axis=0)+1)/(two_train.shape[0]+2)
#print(prob_two_1)
prob_two_0=1-prob_two_1
#print(prob_two_0)

prob_four_one=(four_train.sum(axis=0)+1)/(four_train.shape[0]+2)
#print(prob_four_one)
prob_four_zero=1-prob_four_one
#print(prob_four_zero)

prob_two=two_train.shape[0]/train_X.shape[0]
#print(prob_two)
prob_four=four_train.shape[0]/train_X.shape[0]
#print(prob_four)

def naivebayes(input):
    a=[]
    for i in input:
        log_two=np.log(prob_two)+(np.sum(np.log(prob_two_1[i==1])))+(np.sum(np.log(prob_two_0[i==0])))
        log_four=np.log(prob_four)+(np.sum(np.log(prob_four_one[i==1])))+(np.sum(np.log(prob_four_zero[i==0])))
        a.append(2 if log_two>log_four else 4)
    return np.array(a)
pred_trainX=naivebayes(train_X)
pred_testX=naivebayes(test_X)
#print(pred_trainX)
#print(pred_testX)

train_accuracy=np.mean(pred_trainX==train_Y)*100
test_accuracy=np.mean(pred_testX==test_Y)*100
print("Training Accuracy: ",train_accuracy,"%")
print("Testing Accuracy: ",test_accuracy,"%")

for digit in [2, 4]:
    train_digit_accuracy = np.mean(pred_trainX[train_Y == digit] == digit) * 100
    test_digit_accuracy = np.mean(pred_testX[test_Y == digit] == digit) * 100
    print(digit, ": Training:", train_digit_accuracy,"% , Testing:", test_digit_accuracy,"%")