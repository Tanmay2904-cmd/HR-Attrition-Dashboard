import pandas as pd
import numpy as np
import os
import random
np.random.seed(42)
n=1500
departments=['Sales','Engineering','HR','Finance','Marketing','Operations']
job_roles=['Manager','Senior Executive','Junior Executive','Analyst','Intern']

df = pd.DataFrame({
    'Age': np.random.randint(22,60,n),
    'Department': np.random.choice(departments,n),
    'Job role': np.random.choice(job_roles,n),
    'MonthlyIncome': np.random.randint(20000,100000,n),
    'JobSatisfaction': np.random.randint(1,4,n),
    'OverTime': np.random.choice(['Yes','No'],n),})

attrition_prob = (
    0.40 * (df['MonthlyIncome']<40000).astype(int)+
    0.30 * (df['OverTime']=='Yes').astype(int)+
    0.20 * (df['JobSatisfaction'] <= 2).astype(int))

df['Attrition'] = (attrition_prob > attrition_prob.median()).astype(int)

os.makedirs('data',exist_ok=True)
df.to_csv('data/hr_data.csv',index=False)
print(f"Data saved! Total rows:{len(df)}")
print(f"Attrition rate:{df['Attrition'].mean()*100:.1f}%")
