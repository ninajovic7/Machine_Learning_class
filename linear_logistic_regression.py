import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#from copy import deepcopy
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

plt.rcParams['axes.linewidth']=1.5

col_names = ['name','stamina','attack_value','defense_value','capture_rate','flee_rate','spawn_chance','primary_strength','combat_point']
#data  = pd.read_csv('hw2_data.csv',names =col_names)
data  = pd.read_csv('hw2_data.csv')#,names =col_names)

#(ii)
def plot_ii():
    col_name = ['stamina','attack_value','defense_value','capture_rate','flee_rate','spawn_chance']
    row = 0
    col = 0
    fig,ax = plt.subplots(2,3, figsize=[9,7])
    for i in range(0,len(col_name)):
        ax[col][row].scatter(data[col_name[i]], data["combat_point"],color = 'blue')
        #ax[col][row].scatter(data["combat_point"], data[col_name[i]], color = 'blue')
        ax[col][row].set_ylabel('Combat point', fontsize = 15)
        ax[col][row].set_xlabel(str(col_name[i]), fontsize = 15)
        ax[col][row].tick_params(width = 1, length = 6, labelsize=10)
        p = data["combat_point"].corr(data[col_name[i]])
        ax[col][row].set_title('Pearsons corr = ' + str(round(p,2)), fontsize = 15)
        col = col+1
        if col==2:
            row = row+1
            col = 0

    plt.tight_layout()
    plt.show()

def plot_iii():
    col_name = ['stamina','attack_value','defense_value','capture_rate','flee_rate','spawn_chance']
    row = 0
    col = 0
    fig,ax = plt.subplots(3,5, figsize=[14,7])
    for i in range(0,len(col_name)):
        for j in range(i+1, len(col_name)):

            ax[col][row].scatter(data[col_name[i]], data[col_name[j]],color = 'blue')
            ax[col][row].set_ylabel(str(col_name[j]), fontsize = 10)
            ax[col][row].set_xlabel(str(col_name[i]), fontsize = 10)
            ax[col][row].tick_params(width = 1, length = 5, labelsize=7)
            p = data[col_name[j]].corr(data[col_name[i]])
            ax[col][row].set_title('Pearsons corr = ' + str(round(p,2)), fontsize = 10)
            col = col+1
            if col==3:
                row = row+1
                col = 0

    plt.tight_layout()
    plt.show()

def hot_encoding(col_name,data):
    for i in range(0,len(col_name)):
        unique= pd.unique(data[col_name[i]])
        unique = [x for x in unique if x == x]
        for j in range(0,len(unique)):
            arr = np.zeros(len(data[col_name[i]]))
            arr[np.where(unique[j] == data[col_name[i]])[0]] = 1
            df1 = pd.DataFrame(arr,columns= [unique[j]])
            data= pd.concat([data,df1], axis = 1)
        data = data.drop(col_name[i], axis = 1)
    return data

def compute_W(X,Y):
    X_transp = np.transpose(X)
    try:
        X_inverse = np.linalg.pinv(np.dot(X_transp,X))
    except:
        print('pinv has to be taken')
        X_inverse = np.linalg.pinv(np.dot(X_transp,X))

    W = np.dot(np.dot(X_inverse,X_transp),Y)
    return W

def compute_W_l2_norm(X,Y, lam):
    X_transp = np.transpose(X)
    X_dot = np.dot(X_transp,X)
    try:
        X_inverse = np.linalg.pinv(X_dot +lam*np.identity(len(X_dot)))
    except:
        print('pinv has to be taken')
        X_inverse = np.linalg.pinv(X_dot + lam*np.identity(len(X_dot)))

    W = np.dot(np.dot(X_inverse,X_transp),Y)
    return W

def compute_RSS(X,Y,W):
    XW = np.dot(X, W)
    yxw_t = np.transpose(Y - XW)
    yxw = Y - XW
    RSS = np.dot(yxw_t, yxw)
    return RSS

def regression_with_cross_validation(data,n,lam):
    #data = data.sample(frac=1) #random order
    data.insert(0,'Int',np.ones(len(data))) #to make sure X has 1,... for the intercept
    
    categorical_variables = list(data.columns)
    categorical_variables.remove('name')
    categorical_variables.remove('combat_point')
    
    #categorical_variables = ['Int','attack_value', 'defense_value']
    #categorical_variables = ['Int','attack_value', 'defense_value', 'stamina']
    #categorical_variables = ['Int','attack_value', 'defense_value', 'stamina','capture_rate']
    #categorical_variables = ['Int','stamina','attack_value','defense_value','capture_rate','flee_rate','spawn_chance']

    Y_d = data['combat_point']
    X_d = data[categorical_variables]
    X_d = X_d.iloc[:].values
    Y_d = Y_d.iloc[:].values
    
    rss = []
    rss_l2 = []
    rss_l1 = []
    
    #cross-validation
    p = 0
    print('# fold:      sqrt of RSS   lambda = '+str(lam))
    for t in range(0,n):
        if t == n-1:
            X_test = X_d[p:]
            X_train = X_d[:p]
            Y_test = Y_d[p:]
            Y_train = Y_d[:p]
        if t!=n-1:
            Y_test = Y_d[p:len(Y_d)//n+p]
            Y_train = np.array(list(Y_d[0:p])+ list(Y_d[len(Y_d)//n+p:]))
            X_test = X_d[p:len(X_d)//n+p]
            X_train = np.vstack((X_d[0:p],X_d[len(X_d)//n+p:]))
        p = p+len(data)//n
        
        W_train = compute_W(X_train, Y_train)
        W_train_l2= compute_W_l2_norm(X_train, Y_train,lam)
        
        ##l1 norm
        reg = linear_model.Lasso(alpha=lam)
        reg.fit(X_train, Y_train)
        W_train_l1 = reg.coef_
        
        print(W_train_l2, W_train_l1)

        RSS_l1 = compute_RSS(X_test, Y_test, W_train_l1)
        RSS = compute_RSS(X_test, Y_test, W_train)
        RSS_l2 = compute_RSS(X_test, Y_test, W_train_l2)
        
        rss.append(np.sqrt(RSS))
        rss_l2.append(np.sqrt(RSS_l2))
        rss_l1.append(np.sqrt(RSS_l1))
        
        print(t+1,'          ',np.sqrt(RSS))#, model.ssr)
        #print(t+1,'          ',np.sqrt(RSS_l2))#, model.ssr)
        #print(t+1,'          ',np.sqrt(RSS_l1))#, model.ssr)
        #print(t+1,np.sqrt(RSS_l2))#, model.ssr)
        #print(t+1,np.sqrt(RSS_l1))#, model.ssr)
    
    RSS_avg = np.mean(rss)
    RSS_l2_avg = np.mean(rss_l2)
    RSS_l1_avg = np.mean(rss_l1)
    print('average sqrt of RSS no regularization.:', RSS_avg)
    #print('average RSS with l2:', RSS_l2_avg)
    #print('average RSS with l1:', RSS_l1_avg)

def logistic_regression(data, outcome):
    data = data.sample(frac=1) #random order
    categorical_variables = list(data.columns)
    categorical_variables.remove('name')
    categorical_variables.remove('combat_point')

    Y_d = data[outcome]
    X_d = data[categorical_variables]
    
    mean = Y_d.mean()
    arr = np.zeros(len(Y_d))    
    arr[np.where(mean <= Y_d)[0]] = 1

    Y_test = arr[0:int(np.round(len(arr)*0.2))]
    Y_train = arr[int(np.round(len(arr)*0.2)):]
    
    X_test = X_d[0:int(np.round(len(arr)*0.2))]
    X_train = X_d[int(np.round(len(arr)*0.2)):]

    clf = LogisticRegression(penalty = 'none', max_iter = 1000).fit(X_train, Y_train)
    Y_test_pred = clf.predict(X_test)
    print(len(np.where(Y_test_pred == Y_test)[0])/len(Y_test))
    print(clf.score(X_test, Y_test))
    
    lam = [0.01, 0.1,0.5,1.,2.,10.,100]
    for i in range(0,len(lam)): 
        n=5
        p = 0
        score = []
        for t in range(0,n):
            if t == n-1:
                X_test1 = X_train[p:]
                X_train1 = X_train[:p]
                Y_test1 = Y_train[p:]
                Y_train1 = Y_train[:p]
            if t!=n-1:
                Y_test1 = Y_train[p:len(Y_train)//n+p]
                Y_train1 = np.array(list(Y_train[0:p])+ list(Y_train[len(Y_train)//n+p:]))
                X_test1 = X_train[p:len(X_train)//n+p]
                X_train1 = np.vstack((X_train[0:p],X_train[len(X_train)//n+p:]))
            p = p+len(data)//n
            #print('len ',len(Y_test1))#, Y_train1)
            clf = LogisticRegression(penalty = 'l2', max_iter = 5000, C = 1./lam[i]).fit(X_train1, Y_train1)
            #clf.predict(X_test)
        
            score.append(clf.score(X_test1, Y_test1))
            #print(t+1,clf.score(X_test, Y_test))
        clf = LogisticRegression(penalty = 'l2', max_iter = 5000, C = 1./lam[i]).fit(X_train, Y_train)
        print('lambda: ',lam[i], ',  acc: ',np.mean(score),', acc on test: ',clf.score(X_test, Y_test))


###Running the code
outcome= ['combat_point']
categorical_var= ['primary_strength']

data1 = hot_encoding(categorical_var, data)
#logistic_regression(data1,outcome)

lam = [0.1]#,0.5,1.,2.,10.,100]
for i in range(0,len(lam)):
    data1 = hot_encoding(categorical_var, data)
    regression_with_cross_validation(data1,5,lam[i])


#plot_ii()
#plot_iii()


