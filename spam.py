import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data=pd.read_csv(r"D:\VSCode\demo2\spam\combined_data.csv",encoding="latin-1")

print(data.columns)
data = data.iloc[:, 0:2]
data.columns=['lables','text']


#converting to numbers

x=data['text']
y=data['lables']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)    #statifying split this equales spam and ham ratio

vectorizer=CountVectorizer(stop_words='english')
x_train_vec=vectorizer.fit_transform(x_train)    #.fit-transform learns vocabulary and converts into numbers
x_test_vec=vectorizer.transform(x_test)        #converts using same vocabulary

model=MultinomialNB()    #checks probability of spam word
model.fit(x_train_vec, y_train)


y_pred=model.predict(x_test_vec)
print("accuracy:",accuracy_score(y_test,y_pred))

msg=[]
str=input("enter message:")
msg.append(str)

msg_vec=vectorizer.transform(msg)
prediction=model.predict(msg_vec)

prediction=model.predict(msg_vec)

if prediction[0]==1:
    print("spam message")
else:
    print("not spam")

