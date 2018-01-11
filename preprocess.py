import pandas as pd

df = pd.read_csv('./Data/train_1.csv', nrows = 100);
df['clicks'] = df['click'];
df['hours'] = df['hour'] % 100;
df['month'] = ((df['hour'] /100) % 100).astype(int);
df.drop('id', axis = 1, inplace = True);
df.drop('click', axis = 1, inplace = True);
df.drop(df.columns[0], axis = 1, inplace = True);
df.to_csv('train.csv');