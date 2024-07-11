import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('Phishing_Email.csv')
print(df.head())

print(df.isna().sum())
df = df.dropna()
print(df.isna().sum())

email_type_counts = df['Email Type'].value_counts()
print(email_type_counts)

Safe_Email = df[df["Email Type"]== "Safe Email"]
Phishing_Email = df[df["Email Type"]== "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])
print(Safe_Email.shape,Phishing_Email.shape)

#  create a new Data with the balanced E-mail types
Data= pd.concat([Safe_Email, Phishing_Email], ignore_index = True)
print(Data.head())

dta=pd.DataFrame(Data)
csv_file_path='data.csv'
dta.to_csv(csv_file_path, index=False)
print(f"Data has been written to {csv_file_path}")