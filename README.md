# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Clean your data and divide it into two groups: one for training and one for testing.
2. Pick your variables (features) and set the rules (like regularization) to keep the model from getting too distracted by noise.
3. Let the model "learn" from the training data by minimizing errors (the loss function).
4. Check the model's accuracy on the test data; if it's not performing well, tweak your settings and try again.
5. Use the final model to forecast new outcomes and look at the coefficients to see which variables matter most.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Avantika Krishnadas Kundooly
RegisterNumber: 212224040040 
```
## Head values:
```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```

## Output:
<img width="1007" height="177" alt="image" src="https://github.com/user-attachments/assets/8a5a6942-34f6-4916-9034-0037a298a791" />


## Salary data:
```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```

## Output:
<img width="882" height="175" alt="image" src="https://github.com/user-attachments/assets/0a49b489-1c09-4092-b199-f0d306fb4a30" />


## Checking Null function:
```
d1.isnull().sum()
```

## Output:
<img width="251" height="244" alt="image" src="https://github.com/user-attachments/assets/da314da0-5598-459f-bc2e-b20ee02aefd7" />


## Duplicate data:
```
d1.duplicated().sum()
```

## Output:
<img width="87" height="29" alt="image" src="https://github.com/user-attachments/assets/f433fdf0-35d1-4c42-89b9-6bc8530bd648" />

## Data status:
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```

## Output:
<img width="827" height="361" alt="image" src="https://github.com/user-attachments/assets/af9a0e07-aa8a-4c7c-83e9-397de34be2cb" />


## Values of X:
```
x=d1.iloc[:, : -1]
x
```

## Output:
<img width="795" height="359" alt="image" src="https://github.com/user-attachments/assets/5eab5d2a-72c0-4543-a51b-e5cea87c417c" />

## Values of Y:
```
y=d1["status"]
y
```

## Output:
<img width="376" height="206" alt="image" src="https://github.com/user-attachments/assets/514396e9-0c66-4ef9-91c6-541d60c079ac" />

## Prediction values:
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```

## Output:
<img width="632" height="75" alt="image" src="https://github.com/user-attachments/assets/4657efcc-76c5-4a64-ab8e-c39c9aed82bf" />

## Accuracy value:
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

## Output:
<img width="217" height="20" alt="image" src="https://github.com/user-attachments/assets/481664c6-2f6e-4299-bf6e-53268e92c268" />

## Confusion matrix:
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```

## Output:
<img width="319" height="41" alt="image" src="https://github.com/user-attachments/assets/581e4a4e-ff12-490d-a554-942e182c9cd3" />

## Classification report:
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```

## Output:
<img width="437" height="149" alt="image" src="https://github.com/user-attachments/assets/6601b94e-e059-4085-ade3-2085abda5713" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
