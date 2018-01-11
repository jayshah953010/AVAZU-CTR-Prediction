# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

train = pd.read_csv('./train.csv')


sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)
train.device_type.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Device Type')


zero = sum(train['click'] == 0)
one = sum(train['click'] == 1)
zero_count = train.loc[train['click'] == 0, 'device_type'].value_counts() / zero * 100
one_count = train.loc[train['click'] == 1, 'device_type'].value_counts() / one * 100
# Bar width
width = 0.4
one_count.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Clicked', rot=0)
zero_count.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Not clicked', rot=0)
plt.legend()
plt.xlabel('Device Type')
plt.ylabel('Percentage')
sns.despine()
plt.show()


train.device_conn_type.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Device Connection Type')


zero = sum(train['click'] == 0)
one = sum(train['click'] == 1)
zero_count = train.loc[train['click'] == 0, 'device_conn_type'].value_counts() / zero * 100
one_count = train.loc[train['click'] == 1, 'device_conn_type'].value_counts() / one * 100
# Bar width
width = 0.4
one_count.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Clicked', rot=0)
zero_count.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Not clicked', rot=0)
plt.legend()
plt.xlabel('Device Connection Type')
plt.ylabel('Percentage')
sns.despine()
plt.show()


train.banner_pos.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Banner Position')


zero = sum(train['click'] == 0)
one = sum(train['click'] == 1)
zero_count = train.loc[train['click'] == 0, 'banner_pos'].value_counts() / zero * 100
one_count = train.loc[train['click'] == 1, 'banner_pos'].value_counts() / one * 100
# Bar width
width = 0.4
one_count.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Clicked', rot=0)
zero_count.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Not clicked', rot=0)
plt.legend()
plt.xlabel('Banner Position')
plt.ylabel('Percentage')
sns.despine()
plt.show()