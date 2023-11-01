import cv2
import numpy as np
from sklearn import svm

with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')
with_mask=with_mask.reshape(800,50*50*3)
without_mask=without_mask.reshape(800,50*50*3)
X=np.r_[with_mask,without_mask]
labels=np.zeros(X.shape[0])
labels[200:]=1.0
names={0:"mask",1:"No mask"}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,labels,test_size=0.20)
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
x_train, x_test, y_train, y_test = train_test_split(X,labels,test_size=0.20)

svm=SVC()
svm.fit(x_train, y_train)
y_pred=svm.predict(x_test)
#x_test=pca.transform(x_test)

accuracy_score(y_test,y_pred)
haar_data=cv2.CascadeClassifier('F:\dataa.xml')
capture=cv2.VideoCapture(0)
data=[]
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img= capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face= img[y:y + h, x:x + w, :]
            face=cv2.resize(face, (50,50))
            face=face.reshape(1,-1)
            pred=svm.predict(face)[0]
            n=names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250,250), 2)
            print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2)== 27 or len(data)>=200:
            break
capture.release()
cv2.destroyAllWindows()
