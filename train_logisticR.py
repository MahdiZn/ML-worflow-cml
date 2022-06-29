
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.model_selection import StratifiedShuffleSplit ,GridSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, f1_score, roc_auc_score



data=pd.read_csv('clean_Rain_austrialia.csv')


# # Split the data
# 
# * Now that the data are encoded and scaled, separate the features (X) from the target (y, RainTomorrow).
# * Split the data into train and test data sets. This can be done using any method, but consider using Scikit-learn's `StratifiedShuffleSplit` to maintain the same ratio of predictor classes.
# 



data.RainTomorrow.value_counts(normalize=True)




feature_cols =  [x for x in data.columns  if  x not in 'RainTomorrow']

# Get the split indexes
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, 
                                          test_size=0.3, 
                                          random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols],data.RainTomorrow))

# Create the dataframes
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'RainTomorrow']

X_test  = data.loc[test_idx, feature_cols]
y_test  = data.loc[test_idx, 'RainTomorrow']


# In[80]:


y_train.value_counts(normalize=True)


# In[81]:


y_test.value_counts(normalize=True)


# # Logistic Regresion
# * without regularization 
# * with regularization penalty (l1,l2)
# * compare accuracy ,precision,

# In[82]:



lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)

pkl_filename='logisticRegression.pkl'
with open(pkl_filename,'wb') as file:
    pickle.dump(lr,file)

with open ('logisticRegression.pkl','rb') as file :
    lr=pickle.load(file)


pred_train=lr.predict(X_train)
score_train=accuracy_score(pred_train,y_train)
pred_test=lr.predict(X_test)
score_test=accuracy_score(pred_test,y_test)
print(f"score_tarin:{score_train}")
print(f"score_test:{score_test}")

#write metrics to.txxt
with open('metricas.txt','w') as file:
    file.write(f"Training accuracy :{score_train}\n")
    file.write(f"Test accuracy :{score_test}\n")
    
    
plt.figure(figsize=(7,7))
cm=confusion_matrix(y_test,pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.savefig('logistic.png',dpi=80)    
### END SOLUTION		