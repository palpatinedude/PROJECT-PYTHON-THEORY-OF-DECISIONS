# Load libraries  

import pandas as pd
import numpy as np
import csv
import math
import time

import researchpy as rp

# For preprocessing the data  
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as pl

# To split the dataset into train and test datasets  
from sklearn.model_selection import train_test_split

# To calculate the metrics of the model  
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# To model the data with different classifiers  
from sklearn.naive_bayes import GaussianNB
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


import warnings
warnings.filterwarnings('ignore')


#start with pre-processing  
#ερωτημα 1  
names=['age','sex','d1','d2','d3','d4','d5','d6','d7','d8','healthyornotliver'] #change the names of the columns  
data = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv', names=names)
data['sex'].replace(['Female','Male'],[1,0],inplace=True)#change the values from female or male to 1 or 0  
print('\n\nChecking the 100 first rows:\n')
print(data.head(100))
time.sleep(1)


#no of columns and no of rows  
print('The number of instances and attributes are : ',data.shape,'\n') #583 instances and 11 attributes  
time.sleep(1)


data.fillna(data.mean(numeric_only=True).round(1), inplace=True) #replace every missing value with the mean  
X=data[['age','sex','d1','d2','d3','d4','d5','d6','d7','d8']] #independent variables  
Y=data[['healthyornotliver']] #dependent variable
scaler = MinMaxScaler(feature_range=(-1,1))
data_columns=['d1','d2','d3','d4','d5','d6','d7','d8'] #columns that need normalization  ,normalization of the specific columns  
data[data_columns]=scaler.fit_transform(data[data_columns])
print('\nThe data after scaling: \n',data)
time.sleep(1)



#ερωτημα3  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False) #Splitting the dataset into the Training set and Test set
scaler = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




#ερωτημα 4
def display(classifier): #display all the informations for each classifier
  score=classifier.score(X_train,Y_train.values.ravel())
  print('Mean accuracy before cross validation:',round(score.mean(),3))
  k_folds = KFold(n_splits = 5)#cross validation 5splits to estimate the skill of  classifier  on unseen data 
  scores = cross_val_score(classifier,X_train, Y_train.values.ravel(), cv = k_folds,scoring='accuracy')
  print('Mean accuracy after cross validation:',round(scores.mean(),3))
  print('Each fold accuracy: ',scores)
  tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
  specificity = tn / (tn+fp)
  accuracy= tn+tp / (tn+tp+fn+fp) 
  precision= tp / (tp+fp)
  sensitivity= tp / (tp+fn)
  gm=math.sqrt((sensitivity*specificity))
  print('Accuracy: ',round(accuracy,3)) # the % of the total correct classidied data over all the data 
  time.sleep(1)
  print("Precision:",round(precision,3)) #the % of the true positive classified data over the predicted positive  data  
  time.sleep(1)
  print("Recall/Sensitivity:",round(sensitivity,3))  # the % of the predicted positive classified data over the number of true positive data  
  time.sleep(1)
  print('Specificity: ',round(specificity,3)) #the % of the predicted negative classified data over the number of true negative data 
  time.sleep(1)
  print(f'\nConfusion Matrix for classifier :')      
  time.sleep(1)
  print(confusion_matrix(Y_test,Y_pred)) #confusion matrix visualizes and summarizes the performance of a classification algorithm. 
  print('\nModel Card: ')
  print(classification_report(Y_test,Y_pred, digits=2)) # is used to measure the quality of predictions from a classification algorithm.
  print('The geometric mean is :',round(gm,3))
  return accuracy,precision,sensitivity,specificity,gm


def calculate_the_best(anylist):#find the highest metrics
    maxitem=0.0
    index=0
    for i,item in  enumerate(anylist):
      if(item>maxitem):
         maxitem=item
         index=i
    return index,maxitem  

  
def bestgamma(bestC): # find optimal gamma based on linear search and best score
  bestscore=0.0
  for i in np.arange(0,11,0.5):
   svm1=svm.SVC(C=bestC,kernel='rbf',gamma=i)
   svm1.fit(X_train, Y_train.values.ravel()) 
   Y_pred = svm1.predict(X_test)
   score=svm1.score(X_train,Y_train.values.ravel())
   if(score>bestscore):
      bestscore=score
      bestgamma=i

  print('Best score :' ,round(bestscore,2),'using gamma: ',bestgamma)   
  return  bestgamma


def Cbest_Kbest(choice)  :# Calculating the optimal C or K based on linear search  and best score 
 bestscore=0.0 
 if choice==1: 
   for i in range(1,1201,5):
     svm_clf=svm.SVC(C=i,kernel='linear') #create svm classifier using linear kernel as said to the description to find optimal C  
     svm_clf.fit(X_train,Y_train.values.ravel())
     Y_pred = svm_clf.predict(X_test)
     score=svm_clf.score(X_train,Y_train.values.ravel())
     if(score>bestscore):
      bestscore=score
      bestC=i

   print('Best score :' ,round(bestscore,2),'using C: ',bestC)
   return bestC

 if choice==0:
   bestscore=0.0
   for i in range(3,16):
     knn_classifier = KNeighborsClassifier(n_neighbors=i) #create knn classifier to find optimal K 
     knn_classifier.fit(X_train, Y_train.values.ravel())
     Y_pred = knn_classifier.predict(X_test)
     score=knn_classifier.score(X_train,Y_train.values.ravel())
     if(score>bestscore):
      bestscore=score
      bestK=i
   print('Best score : ',round(bestscore,2),'using K: ',bestK)
   return bestK

def compare_clf(clf1,clf2,clf3): # compare the classifications models
  models = []
  models.append(('GaussianNB', clf1))
  models.append(('KNN', clf2))
  models.append(('SVM', clf3))
  # evaluate each model in turn
  results = []
  names = []
  scoring = 'accuracy'
  for name, model in models:
    kfold =KFold(n_splits=5)
    cv_results = cross_val_score(model, X_train, Y_train.values.ravel(), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: MEAN ACCURACY:',round(cv_results.mean(),3),' STANDARD DEVIATION:', round(cv_results.std(),3))

def btime(timelist): # find the fastest algorithm
  mintime=timelist[0]
  classifier=['GaussianNB','KNN','SVM']
  for index,time in enumerate(timelist):
    if(time<mintime):
      mintime=time
      keep_index=index     
  return classifier[keep_index],mintime
    
#main
import time
timelist=[]
listacc=[]
listspec=[]
listsens=[]
listprec=[]
listgm=[]
print('----------------------------------------------')
print('START THE NAIVE BAYES CLASSIFIER: \n')
start_time = time.time() 
gaussian=GaussianNB() #create Gaussian Naive Bayes Classifier  
gaussian.fit(X_train,Y_train.values.ravel()) #fit and train the data  
Y_pred=gaussian.predict(X_test)              #predict on the test data  
accuracygn,precisiongn,sensitivitygn,specificitygn,gmgn=display(gaussian)
listacc.append(accuracygn)
listprec.append(precisiongn)
listsens.append(sensitivitygn)
listspec.append(specificitygn)
listgm.append(gmgn)
time1=round((time.time() - start_time),3)
print("Execution time: " + str(time1) + ' ms\n')#calculate the time it takes to train the classifier and display the results
timelist.append(time1)

time.sleep(10)

print('----------------------------------------------')
print('START THE KNN CLASSIFIER : \n')
bestK=Cbest_Kbest(0)
time.sleep(1)
start = time.time()
knn=KNeighborsClassifier(n_neighbors=bestK) # create a knn classifier
knn.fit(X_train, Y_train.values.ravel()) #train the model using the training sets
Y_pred = knn.predict(X_test) #predict on the test data
accuracyknn,precisionknn,sensitivityknn,specificityknn,gmknn=display(knn)
listacc.append(accuracyknn)
listprec.append(precisionknn)
listsens.append(sensitivityknn)
listspec.append(specificityknn)
listgm.append(gmknn)
time2=round((time.time() - start),3)
print("Execution time: " + str(time2) + ' ms\n')#calculate the time it takes to train the classifier and display the results
timelist.append(time2)

time.sleep(10)

print('----------------------------------------------')
print('START THE SVM CLASSIFIER :\n ')
bestC=Cbest_Kbest(1)
time.sleep(1)
start_time = time.time()
svm = svm.SVC(C=bestC,kernel='rbf',gamma=bestgamma(bestC)) # create a svm classifier,radias basis function Kernel 
svm.fit(X_train, Y_train.values.ravel()) #train the model using the training sets
Y_pred = svm.predict(X_test) #predict on the test data
accuracysvm,precisionsvm,sensitivitysvm,specificitysvm,gmsvm=display(svm)
listacc.append(accuracysvm)
listprec.append(precisionsvm)
listsens.append(sensitivitysvm)
listspec.append(specificitysvm)
listgm.append(gmsvm)
time3=round((time.time() - start_time),3)
print("Execution time: " + str(time3) + ' ms\n\n')#calculate the time it takes to train the classifier and display the results
timelist.append(time3)

time.sleep(10)



print('--------COMPARE CLASSIFICATION MODELS--------')
name,mintime=btime(timelist) #find the fastest classifier 
compare_clf(gaussian, knn, svm)   
print(f'The fastest algorithm was: {name} with time={mintime}') # we see which algorithm is the fastest
ac,maxac=calculate_the_best(listacc) 
if(ac==0):
    print(f'The highest accuracy:',round(maxac,2),' classifier: GAUSSIANNB',)
if(ac==1):
    print(f'The highest accuracy:',round(maxac,2),' classifier: KNN')
if(ac==2):
    print(f'The highest accuracy:',round(maxac,2),' classifier: SVM') 
pre,maxpre=calculate_the_best(listprec)
if(pre==0):
    print(f'The highest precision:',round(maxpre,2),' classifier: GAUSSIANNB')
if(pre==1):
    print(f'The highest precision:',round(maxpre,2),' classifier: KNN')
if(pre==2):
    print(f'The highest precision:',round(maxpre,2),' classifier: SVM') 
se,maxse=calculate_the_best(listsens)
if(se==0):
    print(f'The highest sensitivity:',round(maxse,2),' classifier: GAUSSIANNB')
if(se==1):
    print(f'The highest sensitivity:',round(maxse,2),' classifier: KNN')
if(se==2):
    print(f'The highest sensitivity:',round(maxse,2),' classifier: SVM') 
sp,maxsp=calculate_the_best(listspec)
if(sp==0):
    print(f'The highest specificity:',round(maxsp,2),' classifier: GAUSSIANNB')
if(sp==1):
    print(f'The highest specificity:',round(maxsp,2),' classifier: KNN')
if(sp==2):
    print(f'The highest specificity:',round(maxsp,2),' classifier: SVM')  
gm,maxgm=calculate_the_best(listgm)
if(gm==0):
    print(f'The highest geometric mean:',round(maxgm,2),' classifier: GAUSSIANB')
if(gm==1):
    print(f'The highest geometric mean:',round(maxgm,2),' classifier: KNN')
if(gm==2):
    print(f'The highest geometric mean:',round(maxgm,2),' classifier: SVM') 

time.sleep(10)



#ερωτημα 5
def t_test():# statistical t tests find the most important attributes by calculating the p-value
 for i in X:
   summary,results=rp.ttest(group1=data[i][data['healthyornotliver']==1],
                           group1_name='healthy liver',
                           group2=data[i][data['healthyornotliver']==2],
                           group2_name='unhealthy liver')
   print('\n\n----------------------------------------------------------')                         
   print(f'Description of the column {i} :\n')                        
   print(summary,results,'\n')  
   time.sleep(5)
 

t_test()
print('Notice that d1,d2,d3,d4,d7 have the least p-value so we can reject the null hypothesis that  data came from the same distribution') 


X.drop('age',inplace=True,axis=1)
X.drop('sex',inplace=True,axis=1)
X.drop('d6',inplace=True,axis=1)
X.drop('d5',inplace=True,axis=1)
X.drop('d4',inplace=True,axis=1)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.2, shuffle=False) #Splitting the dataset into the Training set and Test set
scaler = MinMaxScaler(feature_range=(-1,1)).fit(X1_train)
X1_train = scaler.transform(X1_train)
X1_test = scaler.transform(X1_test)


print('----------------------------------------------')
print('GAUSSIANNB CLASSIFIER :\n ')
time.sleep(1)
start_time = time.time()
gaussian1=GaussianNB() #create Gaussian Naive Bayes Classifier  
gaussian1.fit(X1_train,Y1_train.values.ravel()) #fit and train the data  
Y1_pred=gaussian1.predict(X1_test)              #predict on the test data 
score=gaussian1.score(X1_train,Y1_train.values.ravel())
print('Mean accuracy before cross validation:',round(score.mean(),3))
k_folds = KFold(n_splits = 5)#cross validation 5splits to estimate the skill of  classifier  on unseen data 
scores = cross_val_score(gaussian1,X1_train, Y1_train.values.ravel(), cv = k_folds,scoring='accuracy')
print('Mean accuracy after cross validation:',round(scores.mean(),3))
print('Each fold accuracy: ',scores)
tn, fp, fn, tp = confusion_matrix(Y1_test, Y1_pred).ravel()
specificity = tn / (tn+fp)
accuracy= tn+tp / (tn+tp+fn+fp) 
precision= tp / (tp+fp)
sensitivity= tp / (tp+fn)
gm=math.sqrt((sensitivity*specificity))
print('Accuracy: ',round(accuracy,3)) # the % of the total correct classidied data over all the data 
time.sleep(1)
print("Precision:",round(precision,3)) #the % of the true positive classified data over the predicted positive  data  
time.sleep(1)
print("Recall/Sensitivity:",round(sensitivity,3))  # the % of the predicted positive classified data over the number of true positive data  
time.sleep(1)
print('Specificity: ',round(specificity,3)) #the % of the predicted negative classified data over the number of true negative data 
time.sleep(1)
print(f'\nConfusion Matrix for classifier :')      
time.sleep(1)
print(confusion_matrix(Y1_test,Y1_pred)) #confusion matrix visualizes and summarizes the performance of a classification algorithm. 
print('\nModel Card: ')
print(classification_report(Y1_test,Y1_pred, digits=2)) # is used to measure the quality of predictions from a classification algorithm.
print('The geometric mean is :',round(gm,3))
time4=round((time.time() - start_time),3)
print("Execution time: " + str(time4) + ' ms\n\n')


