import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
NA_VALUES = ["", "NA", "N/A", "null", "None", "nan", -999]
train=pd.read_csv('train.csv',na_values=NA_VALUES)
test=pd.read_csv('test.csv')
train1=train.drop(columns=['id'])
patientid=test['id']
test_filtered=test.drop(columns=['id'])
#build dataframe for our work
#convert numerical from strings 
df=train1.copy()
df["Heart Disease"]=df["Heart Disease"].map({"Presence":1,"Absence":0})

#corelation matrixand heatmap
corr=df.corr()
plt.figure(figsize=(14,10))
heatmap=sns.heatmap(corr, vmin=-1,vmax=1,annot=True,cmap="YlGnBu",
                  annot_kws={"size":8},fmt=".2f")
plt.show()
plt.close()
#now since everything is numeric
#we use simple imputer to fix it 

imputer=SimpleImputer(strategy='median')
df[:]=imputer.fit_transform(df)

#Sex vs Heart Disease
ct = pd.crosstab(df["Sex"], df["Heart Disease"])   # rows: Sex, cols: target
ct.plot(kind="bar")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Sex vs Heart Disease (counts)")
plt.legend(title="Heart Disease")
plt.tight_layout()
plt.show()
plt.close()

#BP and Chorestorol has zero correlation
#we wanna see a scatterplot yet 


#now we want to use the test train split thing
#Out put is Heart Disease Column

#input output split

x_tr_data=df.drop(columns=["Heart Disease"])
y_tr_data=df["Heart Disease"]

#there is nothing else to be done 
#call in test train split 

from sklearn.model_selection import train_test_split
x_tr, x_te , y_tr, y_te=train_test_split(x_tr_data,y_tr_data,test_size=0.1,random_state=20)

#now we choose different models to train it 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#lin=LinearRegression()
logistic=LogisticRegression(max_iter=20000,solver="saga")
rendi=RandomForestClassifier(n_estimators=5000,random_state=20,n_jobs=22)

#import accuray score stuffs 
from sklearn.metrics import accuracy_score

#lin.fit(x_tr, y_tr) #linear fit
#linpred=lin.predict(x_te)

#print("LInear Model Accuray:",accuracy_score(y_te, linpred))

#logstic fit

logistic.fit(x_tr, y_tr) #linear fit
logpred=logistic.predict(x_te)
print("Logistic Model Accuray:",accuracy_score(y_te, logpred))

#RND Classifier
#rendi.fit(x_tr,y_tr)
#repred=rendi.predict(x_te)
#print("RNDF Model Accuray:",accuracy_score(y_te, repred))
#we had Logistic Model Accuray: 0.8817460317460317
#RNDF Model Accuray: 0.8799206349206349
#we move forward with logistic regression

#train on the full tranning set 
logistic.fit(x_tr_data, y_tr_data)
print("full set tranning successful.")
#call in test data for prediciting prob
# probability of Presence (= class 1)
proba = logistic.predict_proba(test_filtered)
p_presence = proba[:, list(logistic.classes_).index(1)]  # robust way (find class 1)

# submission
sub = pd.DataFrame({
    "id": test["id"],
    "Heart Disease": p_presence
})
sub.to_csv("Heart Diseases Submission.csv", index=False)

print(sub.head())