"""
@authors: Fanchon Herman and Cassandre Lepercque
In short: Usage of random Anova on the dataset politeness.
"""

################################
# Packages needded
#################################

from download import download
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sns

################################
# Download datasets
#################################

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

politeness = "http://www.bodowinter.com/uploads/1/2/9/3/129362560/politeness_\
data.csv"
path_politeness = os.path.join(results_dir, "politeness.txt")

download(politeness, path_politeness, replace=False)

df_politeness = pd.read_csv(path_politeness, sep=",", header=0)
print(df_politeness)

# Gender F : female and M :  male
# Conditions pol : polite reponses and inf : informal reponses
# Scenarios hypothetical
# Pitch frequencies

################################
# Rename columns
#################################

df_politeness.rename(columns={"attitude": "condition", "frequency": "pitch"},
                     inplace=True)

################################
# Missing values
#################################

df_politeness.isna().sum()
df_politeness.dropna(inplace=True)

# we see that there is a NA-value in the variable pitch,
# so, we decide to erase the raw that correspond to the NA-value.

################################
# Data visualization
#################################

# Gender repartition

plt.figure()
df_politeness.gender.value_counts().plot(kind='pie', labels=["M", "F"])
plt.title("Sexe repartition for the dataset.")
plt.show()

# we just saw that the gender repartition is equal for male and female.

# Condition repartition

plt.figure()
df_politeness.condition.value_counts().plot(kind='pie', labels=["pol", "inf"])
plt.title("Conditions repartition for the dataset.")
plt.show()

# we just saw that the condition repartition is equal for polite and
# informal reponses.

# Boxplot for visualize the data

sns.catplot(x='condition', y="pitch",
            hue='subject', data=df_politeness, kind="violin", legend=False,
            aspect=2)
plt.title("Pitch by conditions for each subjects")
plt.legend(loc='best')
plt.tight_layout()

# we saw that male subjects have lower voices than female subjects.
# furthemore, we have noticed that there are individual variation for each sex.
# indeed, some females got higher values that others.

#####################################################
# Modeling individual means with random intercepts
#####################################################


