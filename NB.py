import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

def data_label(s):
    it={b'ad.':0, b'nonad.':1}
    return it[s]

path='ad_cranfield.data'
data= np.loadtxt(path, dtype=float, delimiter=',', converters={1558:data_label})

x,y=np.split(data,(1558,),axis=1)

train_data,test_data,train_label,test_label=train_test_split(x,y,random_state=1,train_size=0.7,test_size=0.3)

model_nb = GaussianNB()
test_label = test_label.ravel()
model_nb.fit(train_data,train_label)

train_score = model_nb.score(train_data,train_label)
print("trainingset：",train_score)
test_score = model_nb.score(test_data,test_label)
print("testset：",test_score)

predict_label = model_nb.predict(test_data)

m = sm.confusion_matrix(test_label.ravel(), predict_label)
print('confusion matrix：', m, sep='\n')


r = sm.classification_report(test_label.ravel(), predict_label)
print('classification report：', r, sep='\n')

# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc
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

