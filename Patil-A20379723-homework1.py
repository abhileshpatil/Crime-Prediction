
# coding: utf-8

# In[144]:

import pandas as pd
import numpy as nm
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_score


# <H2><B>1. Decision Trees</B></H2>

# <h><b> percentage of positive and negative instances in the dataset<b></h>

# In[145]:

# Decision Trees
# percentage of positive and negative instances in the dataset?

df=pd.read_csv("D:\ML project\Crime Prediction Data\communities-crime-clean.csv")      # Load clean dataset
pi=0
fi=0
i=0
highCrime=[]
while(i<df.ViolentCrimesPerPop.size):
    if(df.ViolentCrimesPerPop[i]>0.1):
        highCrime.append(True)
        i=i+1
        pi=pi+1
    else:
        highCrime.append(False)
        i=i+1
        fi=fi+1

positive=(pi/df.ViolentCrimesPerPop.size)*100                 # Calculate percentage of positive instances in dataset
negative=(fi/df.ViolentCrimesPerPop.size)*100                 # Calculate percentage of negative instances in dataset
print('Positive',positive,'\n','Negative',negative)
demo=df.ViolentCrimesPerPop

df=df.drop('communityname',axis=1)
df=df.drop('state',axis=1)
df=df.drop('fold',axis=1)
df=df.drop('ViolentCrimesPerPop',axis=1)


# <h><b> DecisionTreeClassifier to learn a decision tree to predict highCrime on dataset</b></h>

# <h><b>b i- Find training accuracy, precision and recall for tree</b></h>

# In[146]:

#Decision Tree
#Use DecisionTreeClassifier to learn a decision tree to predict highCrime
# Find training accuracy, precision, and recall for this tree? 

clf=tree.DecisionTreeClassifier()
clf = clf.fit(df, highCrime)
predict_clf=clf.predict(df)

print('Accuracy',metrics.accuracy_score(highCrime,predict_clf))
print('Precision Score',metrics.precision_score(highCrime, predict_clf,average='weighted'))
print('Recall',metrics.recall_score(highCrime,predict_clf,average='weighted'))


# <h><b>b ii-main features used for classification<b><h>

# In[202]:

sfeatures=sorted(zip(df.columns,clf.feature_importances_),key=lambda x: x[1], reverse=True)

print("\n")
print("main features used for classification")
print(pd.DataFrame(sfeatures[:10]))


# <p>The feature selection makes sense because it helps to select subset of optimal features which helps to split the decision tree in high crime and low crime in best way</p>

# <h><b>c i:-   10-fold cross-validation accuracy, precision, and recall<b><h>

# In[173]:

#Decision Tree
#10 Fold Cross-Validation 
#scores = cross_val_score(clf, df, highCrime, cv=10,scoring='accuracy')
#print(scores)

print('Accuracy',cross_val_score(clf, df, highCrime, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(clf, df, highCrime,cv=10, scoring='precision'))
print('Recall',cross_val_score(clf, df, highCrime,cv=10, scoring='recall'))

print("mean of 10 cross cv accuracy is",nm.mean(cross_val_score(clf, df, highCrime, cv=10, scoring='accuracy')))


# <h><b>c ii:- Why are they different from the results in the previous test? </b></h>

# <p>In 10 Fold cross validation, the original sample is randomly partioned into 10 equal sub samples. Each time the algorithm is run, out of 10 sub samples, a single sub sample is retained as validation data for testing the model and the remaining 9 sub samples are used as training data. So the algorithm is run 10 times, Every time it retains different sub sample exactly once as the validation data and remaining sub samples are used as training data. So each data instance is used as training data 9 times and validation data 1 time.</p>
# <p>And when we run algorithm on a dataset without cross-validation to find training accuracy, precision and recall it consider whole dataset as a training data.</p>
# <p>So we get different values for 10 fold cross-validation accuracy, precision and recall than the results we got previously when run the algorithm on whole training dataset once</p>

# <h2>2. Linear Classification</h2>

# <h><b>a Use GaussianNB to learn a Naive Bayes classifier to predict highCrime.</b></h>

# <h><b>a i- 10-fold cross-validation accuracy, precision, and recall</b></h>

# In[174]:

#Linear Classification
#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
#gnb.fit(df, highCrime)
#print(gnb.predict(df))
pred = gb.fit(df, highCrime).predict(df)
print(pred)

print('Accuracy',cross_val_score(gb, df, highCrime, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(gb, df, highCrime,cv=10, scoring='precision'))
print('Recall',cross_val_score(gb, df, highCrime,cv=10, scoring='recall'))
print("mean of 10 cross accuracy is",nm.mean(cross_val_score(gb, df, highCrime, cv=10, scoring='accuracy')))


# <h><b>a ii- 10 most predictive features</b></h>

# In[203]:

# Linear Classification
# 10 most predictive features
i=0
pos=[]
neg=[]
fin=[]
while(i<df.values.shape[1]):
    j=0
    while(j<df.values.shape[0]):
        if(highCrime[j]==True):
            pos.append(df.values[j][i])
            j=j+1
        else:
            neg.append(df.values[j][i])
            j=j+1
    pos_m=nm.mean(pos)
    neg_m=nm.mean(neg)
    pos_sd=nm.std(pos)
    neg_sd=nm.std(neg)
    formu=abs((pos_m-neg_m))/(pos_sd+neg_sd)
    #fin.append()
    fin.append(formu)
    pos=[]
    neg=[]
    i=i+1

pfeatures=sorted(zip(df.columns,fin),key=lambda x: x[1], reverse=True)

print("10 most predictive features:")
pfeatures[:10]


# <h><b>a iii- How do these results compare with your results from decision trees, above</b></h>

# <p>The overall results of Gaussian naive bayes we calculated gives better values than the result we calculated using decision tree classifier algorithm</p>
# <p>The mean of 10 fold cross-validation accuracy of gaussian naive bayes we got is 0.761608040201 , which is higher than the mean of 10 fold cross-validation accuracy of decision tree classifier we got 0.726989949749</p>
# <p>So as per the results we calculated we got higher accuracy for gaussian naive bayes compared to decision tree classifier</p>

# <h><b>b Use LinearSVC to learn a linear Support Vector Machine model to predict highCrime.</b></h>

# <h><b>b i- 10-fold cross-validation accuracy, precision, and recall</b></h>

# In[178]:

#Linear Classification
#Using LinearSVC

from sklearn import svm
sv = svm.SVC(kernel='linear')
sv=sv.fit(df, highCrime)
afw=sv.coef_

print('Accuracy',cross_val_score(sv, df, highCrime, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(sv, df, highCrime,cv=10, scoring='precision'))
print('Recall',cross_val_score(sv, df, highCrime,cv=10, scoring='recall'))

print("10 fold cross-validation accuracy mean",nm.mean(cross_val_score(sv, df, highCrime, cv=10, scoring='accuracy')))


# <h><b>b ii- most predictive features</b></h>

# In[152]:

LC_pfeatures=sorted(zip(df.columns,afw[0]),key=lambda x: x[1], reverse=True)

print("\n")
print("10 most predictive features:")
LC_pfeatures[:10]


# <h><b>b iii. How do these results compare with your results from decision trees, above?</b></h>

# <p>The overall results of SVM algorithm we calculated gives better values than the result we calculated using decision tree classifier algorithm</p>
# <p>The 10 fold cross validation accuracy, precision and recall we got from svm algorithm is higher than the accuracy and precision value we got from decision tree classifier</p>
# <p>The mean of 10 fold cross-validation accuracy of SVM we got is 0.804753768844
#  , which is higher than the mean of 10 fold cross-validation accuracy of decision tree classifier we got 0.726989949749</p>

# <h2><b>3. Regression</b></h2>

# <h><b>a-  Use LinearRegression to learn a linear model directly predicting the crime rate per capita</b></h>

# <h><b>a i- 10-fold cross-validation, what is the estimated mean-squared-error (MSE)</b></h>

# In[220]:

# Regression
# UseLinearRegression
# Estimated mean-squared-error(MSE) Using 10-fold cross-validation

from sklearn import linear_model,cross_validation
from sklearn.metrics import mean_squared_error
Lreg = linear_model.LinearRegression()
Lreg=Lreg.fit(df,highCrime)

Lreg.coef_
Lreg.intercept_

crossval = cross_validation.cross_val_score(Lreg, df, highCrime, scoring='neg_mean_squared_error', cv=10,)
print("MSE using 10-fold cross-validation",crossval)
mean_crossval=nm.mean(crossval)
print("mean of MSE using 10-fold cross-validation",mean_crossval)


# <h><b>a ii.	What is the MSE on the training set </b></h>

# In[154]:

# Regression
# Mean Square Error on the training set
predict_Lreg=Lreg.predict(df)
print("Mean Square Error",mean_squared_error(highCrime,predict_Lreg))

# iii.	What features are most predictive of a high crime rate? A low crime rate?


# <h><b>a iii. most predictive of a high crime rate and A low crime rate? </b></h>

# In[184]:

LR_pfeaturesH=sorted(zip(df.columns,Lreg.coef_),key=lambda x: x[1], reverse=True)
print("\n")
print("Most Predictive features of a highCrime rate:")
print(pd.DataFrame(LR_pfeaturesH[:10]))

LR_pfeaturesL=sorted(zip(df.columns,Lreg.coef_),key=lambda x: x[1])
print("\n")
print("Most Predictive features of a Low Crime rate:")
print(pd.DataFrame(LR_pfeaturesL[:10]))


# <h><b>b use Ridge regression</b></h>

# <h><b>b i- Estimated MSE of the model under 10-fold CV</b></h>

# In[195]:

#regression
#useRidgeCV

from sklearn.linear_model import Ridge,RidgeCV
rrcv = linear_model.RidgeCV(alphas=[10, 1, 0.1, 0.01, 0.001])
rrcv.fit(df, highCrime)

sco = cross_validation.cross_val_score(rrcv, df, highCrime, scoring='neg_mean_squared_error', cv=10,)
print("Estimated MSE of the model under 10-fold CV",sco)               # Estimated MSE of the model under 10-fold CV



# <h><b>b ii- MSE on the training set </b></h>

# In[196]:

predict_rrcv=rrcv.predict(df)                                          # Mean square error
print("Mean square error",mean_squared_error(highCrime,predict_rrcv))


# <h><b>b iii- What is the best alpha</b></h>

# In[197]:

rrcv = linear_model.RidgeCV(alphas=[10, 1.0, 0.1, 0.01, 0.001])     # Best alpha
rrcv.fit(df,highCrime)
print("Best alpha ",rrcv.alpha_)


# <p>The alphas values quite greater than 1.0, as in this problem is 10, results in underfiting.</p>
# <p>The alpha value 1.0 keeps the model balanced from overfitting.Hence,the alpha value 1.0 is best value to reduce the amount of overfitting.</p>

# <h><b>c. use polynomial features</b></h>

# <h><b>c i- estimated MSE of the model under 10-fold CV</b></h>

# In[227]:

# Polynomial Features 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
poly = PolynomialFeatures(degree=2)

xyz = linear_model.LinearRegression()
pipeline=Pipeline([('polynomial_features',poly),('linear_regression',xyz)])

 
poly_score = cross_validation.cross_val_score(pipeline, df, highCrime, scoring='neg_mean_squared_error', cv=10,)
print("Estimated MSE of the model under 10-fold CV",poly_score)
print("Mean of Estimated MSE of the model under 10-fold CV",nm.mean(poly_score))


# <h><b>c ii- MSE on the training set </b></h>

# In[228]:

pipeline=pipeline.fit(df,highCrime)
pf_predict=pipeline.predict(df)
print("MSE",metrics.mean_squared_error(highCrime,pf_predict))


# <h><b>c iii- Does this mean the quadratic model is better than the linear model for this problem?</b></h>

# <p>The mean of 10 fold cross validation MSE for linear model is -0.131115700339 </p>
# <p>The mean of 10 fold cross validation MSE for quadratic model is -1.10540237834  </p>
# <p>so as the value of mean for MSE is less for quadratic it is better</p>

# <h2><b> 4.	Dirty Data</b></h2>

# In[226]:

# Dirty Data

dirtydf=pd.read_csv("D:\ML project\Crime Prediction Data\communities-crime-full.csv")

dirtydf=dirtydf.replace('?', nm.nan)
dirtydf=dirtydf.dropna()

highCrime1=[]
for val in dirtydf["ViolentCrimesPerPop"]:
    if float(val)>0.1:
        highCrime1.append(True)
    else:
        highCrime1.append(False)

dirtydf=dirtydf.drop('communityname',axis=1)
dirtydf=dirtydf.drop('state',axis=1)
dirtydf=dirtydf.drop('fold',axis=1)
dirtydf=dirtydf.drop('community',axis=1)
dirtydf=dirtydf.drop('county',axis=1)
dirtydf=dirtydf.drop('ViolentCrimesPerPop',axis=1)

dclf=tree.DecisionTreeClassifier()
dclf = dclf.fit(dirtydf, highCrime1)
predict_dclf=dclf.predict(dirtydf)

print('Accuracy',metrics.accuracy_score(highCrime1,predict_dclf))
print('Precision Score',metrics.precision_score(highCrime1, predict_dclf,average='weighted'))
print('Recall',metrics.recall_score(highCrime1,predict_dclf,average='weighted'))

print("\n")
print("10 Fold Cross Validation")
print('Accuracy',cross_val_score(dclf, dirtydf, highCrime1, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(dclf, dirtydf, highCrime1,cv=10, scoring='precision'))
print('Recall',cross_val_score(dclf, dirtydf, highCrime1,cv=10, scoring='recall'))

sfeatures1=sorted(zip(dirtydf.columns,dclf.feature_importances_),key=lambda x: x[1], reverse=True)
print("The mean of 10 fold CV Accuracy",nm.mean(cross_val_score(dclf, dirtydf, highCrime1, cv=10, scoring='accuracy')))
print("\n")
print("main features used for classification")
print(pd.DataFrame(sfeatures1[:10]))


# <p>The mean of 10 fold cross-validation for decision tree classifier for clean data is 0.726989949749</p>
# <p>The mean of 10 fold cross-validation for decision tree classifier for full dataset is 0.845920745921</p>
# <p>So dirty dataset is better because we got better results</p>
# 

# <h2><b>5. Teams </b></h2>

# <h><b>a i- Experiment with two learning methods</b></h>

# In[162]:

# Module 5 
# perceptron
# Clean Dataset
perceptron_reg = linear_model.Perceptron()
perceptron_reg.fit(df, highCrime)
print("On clean dataset:")
print('Accuracy',cross_val_score(perceptron_reg, df, highCrime, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(perceptron_reg, df, highCrime,cv=10, scoring='precision'))
print('Recall',cross_val_score(perceptron_reg, df, highCrime,cv=10, scoring='recall'))

print("\n")
print("10 most predictive features")
perceptron_pfeatures=sorted(zip(df.columns,perceptron_reg.coef_[0]),key=lambda x: x[1], reverse=True)
print(perceptron_pfeatures[:10])

# Full dataset
print("\n")
print("On full dataset:")
perceptron_reg_fdata = linear_model.Perceptron()
perceptron_reg_fdata.fit(dirtydf, highCrime1)
print('Accuracy',cross_val_score(perceptron_reg_fdata, dirtydf, highCrime1, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(perceptron_reg_fdata, dirtydf, highCrime1,cv=10, scoring='precision'))
print('Recall',cross_val_score(perceptron_reg_fdata, dirtydf, highCrime1,cv=10, scoring='recall'))


print("\n")
print("10 most predictive features")
perceptron1_pfeatures=sorted(zip(dirtydf.columns,perceptron_reg_fdata.coef_[0]),key=lambda x: x[1], reverse=True)
print(perceptron1_pfeatures[:10])


# <h><b>a i- </b></h>

# In[163]:

# Module 5 
# Logistic Regression
# Clean dataset
Logistic_reg = linear_model.LogisticRegression()
Logistic_reg.fit(df, highCrime)

print("On clean Dataset")
print('Accuracy',cross_val_score(Logistic_reg, df, highCrime, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(Logistic_reg, df, highCrime,cv=10, scoring='precision'))
print('Recall',cross_val_score(Logistic_reg, df, highCrime,cv=10, scoring='recall'))

print("\n")
print("10 most predictive features")
Logistic_pfeatures=sorted(zip(df.columns,Logistic_reg.coef_[0]),key=lambda x: x[1], reverse=True)
print(Logistic_pfeatures[:10])

# Full dataset
print("\n")
print("On full dataset:")
Logistic_reg_fdata = linear_model.LogisticRegression()
Logistic_reg_fdata.fit(dirtydf, highCrime1)
print('Accuracy',cross_val_score(Logistic_reg_fdata, dirtydf, highCrime1, cv=10, scoring='accuracy'))
print('Precision',cross_val_score(Logistic_reg_fdata, dirtydf, highCrime1,cv=10, scoring='precision'))
print('Recall',cross_val_score(Logistic_reg_fdata, dirtydf, highCrime1,cv=10, scoring='recall'))

print("\n")
print("10 most predictive features")
Logistic1_pfeatures=sorted(zip(dirtydf.columns,Logistic_reg_fdata.coef_[0]),key=lambda x: x[1], reverse=True)
print(Logistic1_pfeatures[:10])


# In[164]:

# Module 5 b

