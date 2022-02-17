import numpy as np
import sys
import pandas as pd

def kk_nearest(test,train,k):
    assign = []
    for j in range(0,len(train)):
        eu_d= []
        test_n = []
        for i in range(0,len(test)):
            temp = np.sqrt(np.sum((test[i] - train[j])**2))
            eu_d.append(temp)
            test_n.append(int(test_class[i][0]))
        df = pd.DataFrame({'eu_d':eu_d, 'class':test_n})
        df = df.sort_values(by=['eu_d'])
        df = df[0:k]
        print(df)
        if len(np.where(df['class']==1)[0]) > len(np.where(df['class']==2)[0]):
            assign.append(1)
        else:
            assign.append(2)
    return assign

col_names = ['Age', 'YOP', 'PAN', 'class']
train_p  = pd.read_csv('data_train.csv',names =col_names)
test_p  = pd.read_csv('data_test.csv',names =col_names)
dev_p  = pd.read_csv('data_dev.csv',names =col_names)

test = test_p.iloc[:,0:-1].values
test_class = test_p.iloc[:,3:].values

train = train_p.iloc[:,0:-1].values
train_class = train_p.iloc[:,3:].values

dev = dev_p.iloc[:,0:-1].values
dev_class = dev_p.iloc[:,3:].values

k = [1,3,5,7,9,11,13]

for i in range(0,len(k)):
    assign  =  kk_nearest(dev,train,k[i])
    
    acc = (len(np.where(dev_p["class"] == assign)))/len(assign)
    class1_t = np.where(dev_p["class"] ==1)
    class1 = np.where(np.array(assign) == 1)
    
    class2_t = np.where(dev_p["class"] ==2)
    class2 = np.where(np.array(assign) == 2)
    
    bacc = 0.5 * ( len(np.intersect1d(class1[0], class1_t[0]))/len(class1))+ 0.5 * (len(np.intersect1d(class1[0], class1_t[0]))/len(class1))
    print('k: ',k, 'acc: ', acc, 'bacc: ', bacc)
    print(assign)
