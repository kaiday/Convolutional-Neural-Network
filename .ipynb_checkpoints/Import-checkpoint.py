import pandas as pd

# load the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# add target to DataFrame
df['target'] = data.target

print(df)