# This program is an ID3.0 method applying on the machine learning assignment dataset.
# 
#
#

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

import sklearn.metrics as sm
import graphviz
import pydotplus
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def data_label(s):
    it={b'ad.':0, b'nonad.':1}
    return it[s]

# data preparation
path='ad_cranfield.data'
data= np.loadtxt(path, dtype=float, delimiter=',', converters={1558:data_label})

x,y=np.split(data,(1558,),axis=1)

# randomly split the dataset to 50%:50% of training set and test set
train_data,test_data,train_label,test_label=train_test_split(x,y,random_state=1,train_size=0.7,test_size=0.3)

# model training: parameter"gini" means that it is CART method
model_dt = DecisionTreeClassifier(criterion='entropy',
                                  splitter='best',
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=None,
                                  random_state=None,
                                  min_impurity_decrease=0,
                                  class_weight=None
                                  )

# pre cross validation for model
# model accuracy
score = cross_val_score(model_dt, train_data, train_label.ravel(), cv=5, scoring='accuracy')
print('accuracy score=', score)
print('accuracy mean=', score.mean())

# precision
score = cross_val_score(model_dt, train_data, train_label.ravel(), cv=5, scoring='precision_weighted')
print('precision_weighted score=', score)
print('precision_weighted mean=', score.mean())

# recall
score = cross_val_score(model_dt, train_data, train_label.ravel(), cv=5, scoring='recall_weighted')
print('recall_weighted score=', score)
print('recall_weighted mean=', score.mean())

# f1score
score = cross_val_score(model_dt, train_data, train_label.ravel(), cv=5, scoring='f1_weighted')
print('f1_weighted score=', score)
print('f1_weighted mean=', score.mean())




model_dt.fit(train_data,train_label.ravel())

# model test
train_score = model_dt.score(train_data,train_label.ravel())
print("training set：",train_score)

test_score = model_dt.score(test_data,test_label.ravel())
print("test set：",test_score)



# plot decision tree
dot_data = StringIO()
tree.export_graphviz(model_dt, out_file=dot_data,
                         filled=True, rounded=True,
                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("dtree.pdf")

# plot confusion matrix

predict_label = model_dt.predict(test_data)

m = sm.confusion_matrix(test_label.ravel(), predict_label)
print('confusion matrix：', m, sep='\n')


r = sm.classification_report(test_label.ravel(), predict_label)
print('classification report：', r, sep='\n')

# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(test_label.ravel(), predict_label) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()






