
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit ,GridSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, f1_score, roc_auc_score



data=pd.read_csv('clean_Rain_austrialia.csv')


# # Split the data
# 
# * Now that the data are encoded and scaled, separate the features (X) from the target (y, RainTomorrow).
# * Split the data into train and test data sets. This can be done using any method, but consider using Scikit-learn's `StratifiedShuffleSplit` to maintain the same ratio of predictor classes.
# 

# In[77]:


data.RainTomorrow.value_counts(normalize=True)


# In[79]:


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


# In[84]:


import pickle
pkl_filename='logisticRegression.pkl'
with open(pkl_filename,'wb') as file:
    pickle.dump(lr,file)


# In[83]:


lr_l1=LogisticRegressionCV(cv=4,solver='liblinear',penalty='l1')
lr_l1.fit(X_train,y_train)


# In[85]:


import pickle
pkl_filename='lr_l1Cv.pkl'
with open(pkl_filename,'wb') as file:
    pickle.dump(lr_l1,file)


#  __Compare models__
# 

# In[86]:


from sklearn.metrics import precision_recall_fscore_support,accuracy_score as score

pred_list=[]

model=[lr,lr_l1]
label=['lr','lr_l1']

for lab,mod in zip(label,model):
        pred_list.append(pd.Series(mod.predict(X_test),name=lab))
        
   # metric_list.append(pd.Series(accuracy_score(y_test,mod.predict(X_test)),name=lab))
   # metric_list.append(pd.Series(precision_score(y_test,mod.predict(X_test)),name=lab))
    
    

pred_list = pd.concat(pred_list, axis=1)  


pred_list.head()


# In[87]:


from sklearn.metrics import precision_recall_fscore_support as score


metrics = list()

cm= dict()
for lab in label:

    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, pred_list[lab],average='weighted')
    
    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, pred_list[lab])
    
    
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, pred_list[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy},
                              
                             name=lab))

metrics = pd.concat(metrics, axis=1)


# In[88]:


metrics


# In[90]:


result_metrics = metrics.to_json(orient="columns")


# In[92]:


import json
with open("metrics.json", 'w') as outfile:
        json.dump(result_metrics, outfile)
		
		

fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off')

for ax,lab in zip(axList[:-1], label):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d', cmap='viridis');
    ax.set(title=lab);
    
plt.tight_layout()
plt.savefig("CM_ALL_LR.png",dpi=80)
### END SOLUTION		