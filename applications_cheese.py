"""
@authors: Cherif Amghar
In short: Usage of random Anova on two datasets.
"""

###################
# Packages needded
###################

from download import download
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import numpy as np
import seaborn as sns
import scipy
##################
# Download datasets
##################
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

cheese = "https://stat.ethz.ch/~meier/teaching/data/cheese.dat"
path_cheese = os.path.join(results_dir, "cheese.txt")

download(cheese, path_cheese, replace=False)

df_cheese = pd.read_table(path_cheese, sep="\s+")
print(df_cheese)

################################
# background repartition
#################################

plt.figure()
df_cheese.background.value_counts().plot(kind='pie')
plt.title("Background repartition amongst the dataset.")
plt.show()

# Cheese repartition

plt.figure()
df_cheese.cheese.value_counts().plot(kind='pie')
plt.title("Cheese repartition amongst the dataset.")
plt.show()


plt.figure()
df_cheese.groupby('cheese')['y'].aggregate(lambda x: x.median())\
					.plot(kind='bar')
plt.xlabel('Cheeses')
plt.ylabel('Evaluation')
plt.title('Evaluation for each cheeses')
plt.show()

# MixedLM
md = smf.mixedlm("y ~ background * cheese", data=df_cheese,
				 groups=df_cheese[["rater","backround"]])
mdf = md.fit()
print(mdf.summary())