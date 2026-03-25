# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Load the dataset, select relevant features and target variable, and split the data into training and testing sets.
 
2. Normalize the feature values using standard scaling for better model performance.
 
3. Define an SVM model and specify a grid of hyperparameters (C, kernel, gamma).
  
4. Perform hyperparameter tuning using Grid Search with cross-validation to find the best model.
  
5. Evaluate the optimized model on test data using accuracy, classification report, and confusion matrix visualization.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by:  SUBASHRAM T
RegisterNumber: 212225040430
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class' 

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],              
    'kernel': ['linear', 'rbf'],         
    'gamma': ['scale', 'auto']          
}


grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("="*40)
print("Name: SUBASHRAM T")
print("Reg. No: 212225040430")
print("="*40)
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)
print()
accuracy = accuracy_score(y_test, y_pred)
print("="*40)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("="*40)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

*/
```

## Output:

<img width="1242" height="698" alt="Screenshot 2026-03-25 135350" src="https://github.com/user-attachments/assets/2dffb56a-c5f2-40ba-a0ed-3f617c196957" />
<img width="1242" height="92" alt="Screenshot 2026-03-25 135411" src="https://github.com/user-attachments/assets/3720faf9-2842-4cd6-832e-32fcf1020bab" />
<img width="1244" height="311" alt="Screenshot 2026-03-25 135434" src="https://github.com/user-attachments/assets/030174d0-2d52-44ff-b0b2-22e834d08df2" />
<img width="1225" height="571" alt="Screenshot 2026-03-25 135449" src="https://github.com/user-attachments/assets/ae8ed4e2-38f6-4ccf-b898-d39c2994dd24" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
