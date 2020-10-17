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

# the formula is pitch=int+politeness+gender+(1|subject)+\varepsilon

# idea for the different participant means across conditions
df_politeness.groupby('subject')[['pitch']].aggregate(lambda x: x.mean())

# estimation of the means for each participant including a random intercept
# for each subject
md1 = smf.mixedlm("pitch ~ 1", df_politeness, groups=df_politeness['subject'])
# lmer(pitch ~ (1 | subject), data = d)
md1f = md1.fit()
print(md1f.summary())

# we can see that the mean pitch is 193.025.

#####################################################
# Including fixed effects
#####################################################

# recoding of the variable gender and condition.
df_politeness["gender"].replace({'F': 1, 'M': -1}, inplace=True)
df_politeness["condition"].replace({'inf': 1, 'pol': -1}, inplace=True)

# include a model with condition and gender and a random intercept
# for each subject.
md2 = smf.mixedlm("pitch ~ condition + gender", df_politeness,
                  groups=df_politeness['subject'])
md2f = md2.fit()
print(md2f.summary())

# we can see that the mean pitch is 192.883.
# we noticed that pitch is higher for informal than polite.
# we noticed that pitch is higher for females than males.

# evaluation of the model
# logLikelihood = logLik(res2)
# deviance = -2*logLikelihood[1]; deviance


#####################
# Random Slopes
#####################

# add random slopes

md3 = smf.mixedlm("pitch ~ condition + gender", df_politeness,
                  groups=df_politeness['subject'], re_formula='~condition')
md3f = md3.fit()
print(md3f.summary())
