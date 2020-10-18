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
from statsmodels.formula.api import ols
import numpy as np
import seaborn as sns
import scipy

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

# anova to do

######################
# Testing signifiance
######################


#####################
# Getting p-values
# ~~~~~~~~~~~~~~~~~~

md2b = md2.fit(reml=True)
md3b = md3.fit(reml=True)
print(md3b.summary())

# comparing model outputs

# anova = sm.stats.anova_lm(md2)
# anova_b = sm.stats.anova_lm(md2b)

#####################
# Model comparison
# ~~~~~~~~~~~~~~~~~~

md4 = smf.mixedlm("pitch ~  gender", df_politeness,
                  groups=df_politeness['subject'])
md4f = md4.fit(reml=False)
print(md4f.summary())

md4b = smf.mixedlm("pitch ~ condition + gender", df_politeness,
                   groups=df_politeness['subject'])
md4bf = md4b.fit(reml=False)
print(md4bf.summary())

# anova(md4f, md4bf)

# The likelihood ratio test essentially tells us how much more likely the data
# is under a more complex model than under the simpler model
# D =-2* ln(likelihood for simple model) + 2*ln(likelihood for complex model)
# the distribution of D is approximately chi_2 with df2-df1 degrees of freedom

dev1 = (-2)*md4bf.llf  # deviance complex model
dev0 = (-2)*md4f.llf  # deviance simpler model
dev_diff = dev0 - dev1
print("Stat. of the likehood ratio : %.4f " % (dev_diff))

# params md4bf : 3 fixed + 1 random
# params md4f : 2 fixed + 1 random
# donc df_diff vaut 1

pvalue = 1.0 - scipy.stats.chi2.cdf(dev_diff, 1)

print('Chi square =', np.round(dev_diff, 3), '(df=1)',
      'p=', np.round(pvalue, 6))

# we have compared 2 nested models
# one without condition
# other with condition
# we conclude that inclusion of condition is warranted in our model since
# it significantly improves model fit, χ^2(1)=8.79, p<0.01

#############
# REML vs ML
#############

md5 = smf.mixedlm("pitch ~  condition + gender", df_politeness,
                  groups=df_politeness['subject'])
md5f = md5.fit(reml=False)
md5b = ols('pitch ~ condition + gender', data=df_politeness).fit()

# anova(res5b, res5) # doesn't work!

dev1b = (-2)*md5f.llf
dev0b = (-2)*md5b.llf
dev_diffb = dev0b - dev1b
print(dev_diffb)

# # params md5f : 3 fixed + 1 random
# params md5b : 3 fixed
# donc df_diffb vaut 1

p_value = 1.0 - scipy.stats.chi2.cdf(dev_diffb, 1)

print('Chi square =', np.round(dev_diffb, 3), '(df=1)',
      'p=', np.round(p_value, 6))

# compare the AICs


################
# Item effects
################

# boxplot
plt.figure()
sns.catplot(x='scenario', y="pitch",
            data=df_politeness, kind="box", legend=False,
            aspect=2)
plt.title("Pitch by scenario")
plt.legend(loc='best')
plt.tight_layout()

# mixed-lm

