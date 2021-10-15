import requests
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# define URL and file_path to data
url = 'http://localhost:2000/api'
local_dir = os.path.abspath(os.path.dirname(__file__))
test_data_path = os.path.join(local_dir, 'data/processed/train.csv')
save_path = os.path.join(local_dir, 'data/predictions.csv')
# load data and send request to make prediction
test_data = pd.read_csv(test_data_path, index_col=0)
x_test = test_data.drop(columns=['PassengerId', 'Survived'])
x_test.Age = x_test.Age.fillna(x_test.Age.median())
data = np.array(x_test).tolist()
preds = []
print('requesting predictions...')
for observation in tqdm(data):
	j_data = json.dumps([observation])
	headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
	r = requests.post(url, data=j_data, headers=headers)
	preds.append(r.text)

df_preds = pd.concat([test_data['PassengerId'], pd.Series(preds, name='y_pred')], axis=1)
df_preds.to_csv(save_path, index=False)
print('Data saved to {}'.format(save_path))