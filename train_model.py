import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

df= pd.read_csv('data/hr_data.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
y=df['Attrition']
x=df.drop('Attrition',axis=1)
le = LabelEncoder()
for col in ['Department','Job role','OverTime']:
    x[col]=le.fit_transform(x[col])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model = XGBClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(f"Model Accuracy: {accuracy*100:.1f}%")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/attrition_model.pkl')
print("✅ Model saved!")