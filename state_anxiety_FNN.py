import numpy as np
import pandas as pd
import seaborn as sns 
from google.colab import drive
import matplotlib.pyplot as plt
from scipy import stats

drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/Homework5/data.csv')

features  = data.columns
print(data)
for i in range(1,len(features)):
  data[features[i]] = data[features[i]].replace(np.nan,np.mean(data[features[i]]))

#print(len(features), features)

#for i in range(1,len(features)-2):
#  sns.jointplot(data=data, x="StateAnxiety", y=data[features[i]],kind = "reg")#,hue="species")
#  p = stats.pearsonr(data["StateAnxiety"],data[features[i]])
#  p = np.round(p[0],4)
#  plt.title('p = '+ str(p))
#plt.show()


import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import keras
from keras.callbacks import History
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

beh_f = ['SCL', 'SCRamp', 'SCRfreq', 'HR', 'BVP', 'TEMP', 'ACC', 'IBI',
       'RMSenergy', 'mfcc[1]', 'mfcc[2]', 'mfcc[3]', 'mfcc[4]', 'mfcc[5]',
       'mfcc[6]', 'mfcc[7]', 'mfcc[8]', 'mfcc[9]', 'mfcc[10]', 'mfcc[11]',
       'mfcc[12]', 'zcr', 'voiceProb', 'F0', 'pause_frequency']

###Wrapper method
sfs = SFS(LinearRegression(),
          k_features=5,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)

sfs.fit(data[beh_f], data["StateAnxiety"])
selected_f_wrapper = sfs.k_feature_names_

###Filter Method
beh_fil = ['SCL', 'SCRamp', 'SCRfreq', 'HR', 'BVP', 'TEMP', 'ACC', 'IBI',
       'RMSenergy', 'mfcc[1]', 'mfcc[2]', 'mfcc[3]', 'mfcc[4]', 'mfcc[5]',
       'mfcc[6]', 'mfcc[7]', 'mfcc[8]', 'mfcc[9]', 'mfcc[10]', 'mfcc[11]',
       'mfcc[12]', 'zcr', 'voiceProb', 'F0', 'pause_frequency','StateAnxiety']

X= data[beh_fil]
importances = X.drop("StateAnxiety", axis=1).apply(lambda x: x.corr(X.StateAnxiety))
indices = np.argsort(importances)
selected_f_filer = list(importances[indices].index[0:5])
#print(importances[indices].index)
#print(selected_f_filer)


###FNN
#print(data[list(selected_f)].shape)
history = History()

model = Sequential([
  Dense(units = 32, activation='relu', input_shape=(data[list(selected_f_wrapper)].shape[1],), name="first_hidden_layer"),
  Dense(units = 32, activation='relu', name="second_hidden_layer"), Dropout(0.35),
  Dense(units = 16, activation='relu', name="third_hidden_layer"), Dropout(0.35),
  Dense(units = 6, activation='relu', name="fourth_hidden_layer"), Dropout(0.35),
  Dense(units=1)])

model.compile(optimizer='RMSprop', loss='mse',metrics=['mse'],)
#loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

## Train model

#print(data[list(selected_f_wrapper)])

def FNN_with_cross_val(X_d,Y_d):

  X_d = X_d.iloc[:].values
  Y_d = Y_d.iloc[:].values

  err= []
  p = 0
  n=5

  for t in range(0,n):
    Y_test = Y_d[p:len(Y_d)//n+p]
    Y_train = np.array(list(Y_d[0:p])+ list(Y_d[len(Y_d)//n+p:]))
    X_test = X_d[p:len(X_d)//n+p]
    X_train = np.vstack((X_d[0:p],X_d[len(X_d)//n+p:]))
    p = p+len(data)//n

    model.fit(X_train, Y_train, epochs=50, batch_size=100, callbacks=[history], verbose = False)
    #performance = model.evaluate(X_test, Y_test)
    prediction = model.predict(X_test)
    err.append(np.sum(np.abs(np.array(prediction).flatten() - np.array(Y_test)))/len(Y_test))

  return err

err_wrapper = FNN_with_cross_val(data[list(selected_f_wrapper)],data["StateAnxiety"])
err_filter = FNN_with_cross_val(data[list(selected_f_filer)],data["StateAnxiety"])


print(err_wrapper)
print(err_filter)

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

drive.mount('/content/drive')
val = data["PID"]
linear_features = []

for i in val:
  d1 = pd.read_excel('/content/drive/My Drive/Homework5/EDA_PPT_'+str(i)+'.xlsx')
  d2 = pd.read_excel('/content/drive/My Drive/Homework5/HR_PPT_'+str(i)+'.xlsx')
  #print(d1)
  X1 = d1.iloc[:, 0].values.reshape(-1, 1)
  Y1 = d1.iloc[:, 1].values.reshape(-1, 1)
  regr = linear_model.LinearRegression()
  regr.fit(X1,Y1)
  X2 = d2.iloc[:, 0].values.reshape(-1, 1)
  Y2 = d2.iloc[:, 1].values.reshape(-1, 1)
  regr1 = linear_model.LinearRegression()
  regr1.fit(X2,Y2)
  #print(i, regr.coef_[0][0],regr.intercept_)
  linear_features.append([regr.coef_[0][0],regr.intercept_[0], regr1.coef_[0][0],regr1.intercept_[0]])
  #print(regr.intercept_)

df2 = pd.DataFrame(np.array(linear_features), columns = ['EDA_bias','EDA_int','ER_bias','ER_int'])
#print(df2.iloc[:].values)
#print(df2.columns)
#print(linear_features)

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor

# evaluate model
estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=5)
kfold = KFold(n_splits=5)
results = cross_val_score(estimator, data[list(selected_f_wrapper)], data["StateAnxiety"], cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
