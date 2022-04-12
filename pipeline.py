"""
Example from https://github.com/zykls/folktables

Distribution shift across states
--------------------------------
Each prediction problem in Folktables can be instantiated on data from every US
state. This allows us to not just construct a diverse set of test environments,
but also to use Folktables to study questions aroud distribution shift. For
example, we can train a classifier using data from California and then evaluate
it on data from Michigan.
"""

from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)
mi_data = data_source.get_data(states=["MI"], download=True)
ca_features, ca_labels, _ = ACSIncome.df_to_numpy(ca_data)
mi_features, mi_labels, _ = ACSIncome.df_to_numpy(mi_data)

# Plug-in your method for tabular datasets
model = LogisticRegression()

# Train on CA data
model.fit(ca_features, ca_labels)

# Test on MI data
model.score(mi_features, mi_labels)
