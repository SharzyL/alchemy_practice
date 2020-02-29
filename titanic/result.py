import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize

plt.style.use('dark_background')

ttd = pd.read_csv('../_data/titanic/test.csv')


ttd['title'] = ttd['Name'].str.split('[,.]\s+').str[1]
title_average_age = ttd.groupby('title').mean()['Age']
for i in range(ttd.shape[0]):
    if np.isnan(ttd['Age'][i]):
        ttd['Age'][i] = title_average_age[ttd['title'][i]]

ttd['EmbarkedC'] = (ttd['Embarked'] == 'C')
ttd['EmbarkedQ'] = (ttd['Embarked'] == 'Q')
ttd['isMale'] = (ttd['Sex'] == 'male')
X = ttd[['Pclass', 'isMale', 'Fare', 'Age', 'EmbarkedC', 'EmbarkedQ']]

net_ttd = X.copy().to_numpy(dtype=np.float)
net_ttd = normalize(net_ttd, axis=0, norm='max')


