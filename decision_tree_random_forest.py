import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

data  = pd.read_csv('data1.csv')
col_names = data.keys()

for i in range(1,len(col_names)-1):
    p = data["Hirability"].corr(data[col_names[i]], method='pearson')
    print(col_names[i], p)

print('Decision Tree')

def decision_tree(n):
    regressor = DecisionTreeRegressor(max_depth=n)
    val_test = []
    val_train = []

    for i in range(0,len(data)):
        row = list(np.arange(len(data)))
        row.pop(i)
    
        y_test = data.iloc[i:i+1,-1].values
        x_test = data.iloc[i:i+1,1:len(col_names)-1].values
        x_test = np.nan_to_num(x_test)
        val_test.append(y_test[0])
    
        y_train = data.iloc[row,-1].values
        x_train = data.iloc[row,1:len(col_names)-1].values
        x_train = np.nan_to_num(x_train)

        regressor.fit(x_train, y_train)
        val_train.append(regressor.predict(x_test)[0])
    
    err = np.sum(np.abs(np.array(val_test)  - np.array(val_train)))/len(val_test)
    
    Y = data.iloc[:,-1].values
    X = data.iloc[:,1:len(col_names)-1].values
    X = np.nan_to_num(X)
    regressor.fit(X, Y)
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(regressor, feature_names=data.columns[1:-1],filled = True)
    plt.savefig('tree_best_'+str(n)+'.png')
    

    return err

n = [2,3,4,5,6,7]
ERR = []
for j in range(0,len(n)):

    err = decision_tree(n[j])
    print(n[j],err)
    ERR.append(err)
#nn = np.where(ERR == np.min(ERR))[0]
#N = n[int(nn)]
#print(N)

##plot the best tree
#regressor = DecisionTreeRegressor(max_depth=N)#, random_state = 0)
#Y = data.iloc[:,-1].values
#X = data.iloc[:,1:len(col_names)-1].values
#X = np.nan_to_num(X)
#regressor.fit(X, Y)
        
#fig = plt.figure(figsize=(25,20))
#_ = tree.plot_tree(regressor, feature_names=data.columns[1:-1],filled = True)
                   #class_names=iris.target_names,
                   #filled=True)
#plt.savefig('tree_best_'+str(N)+'.png')


###Random Forest
print('Random Forest')

def random_forest(n,m):
    rf = RandomForestRegressor(n_estimators = n, max_depth = m)#, random_state = 0)
    val_test = []
    val_train = []

    for i in range(0,len(data)):
        row = list(np.arange(len(data)))
        row.pop(i)
    
        y_test = data.iloc[i:i+1,-1].values
        x_test = data.iloc[i:i+1,1:len(col_names)-1].values
        x_test = np.nan_to_num(x_test)
        val_test.append(y_test[0])
    
        y_train = data.iloc[row,-1].values
        x_train = data.iloc[row,1:len(col_names)-1].values
        x_train = np.nan_to_num(x_train)

        rf.fit(x_train, y_train)
        val_train.append(rf.predict(x_test)[0])
        #print(regressor.predict(x_test)[0])
        #print(y_test[0])
    
    err = np.sum(np.abs(np.array(val_test)  - np.array(val_train)))/len(val_test)
    

    return err

n = [10,30,50,100,200,500]
m = [2,3,4,5,6,7]
ERR = []
for j in range(0,len(n)):

    err = random_forest(n[j],m[j])
    print(m[j],n[j],err)
    ERR.append(err)
#nn = np.where(ERR == np.min(ERR))[0]
#N = n[int(nn)]
#print(N)


###Ada Boost

print('Adaboost')

def adaboost(n):
    rf = AdaBoostRegressor(n_estimators=n)#, random_state = 0)
    val_test = []
    val_train = []

    for i in range(0,len(data)):
        row = list(np.arange(len(data)))
        row.pop(i)
    
        y_test = data.iloc[i:i+1,-1].values
        x_test = data.iloc[i:i+1,1:len(col_names)-1].values
        x_test = np.nan_to_num(x_test)
        val_test.append(y_test[0])
    
        y_train = data.iloc[row,-1].values
        x_train = data.iloc[row,1:len(col_names)-1].values
        x_train = np.nan_to_num(x_train)

        rf.fit(x_train, y_train)
        val_train.append(rf.predict(x_test)[0])
        #print(regressor.predict(x_test)[0])
        #print(y_test[0])
    
    err = np.sum(np.abs(np.array(val_test)  - np.array(val_train)))/len(val_test)
    

    return err

n = [2,3,4,5,6,7]#,10,30,50,100,200,500]
ERR = []
for j in range(0,len(n)):

    err = adaboost(n[j])
    print(n[j],err)
    ERR.append(err)
