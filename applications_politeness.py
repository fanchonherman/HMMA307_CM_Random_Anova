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

# we rename the columns attitude and gender in respectively condition and pitch
df_politeness.rename(columns={"attitude": "condition", "frequency": "pitch"},
                     inplace=True)

################################
# Missing values
#################################

df_politeness.isna().sum()
df_politeness.dropna(inplace=True)

# we see that there is a NA-value in the variable pitch
# so, we decide to erase the raw that correspond to the NA-value.

################################
# Data visualization
################################

############################
# Gender repartition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure()
df_politeness.gender.value_counts().plot(kind='pie', labels=["M", "F"])
plt.title("Sexe repartition for the dataset.")
plt.show()

# we just saw that the gender repartition is  pretty equal for male and female
# indeed , we erase a raw that correspond to a NA-value

############################
# Condition repartition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure()
df_politeness.condition.value_counts().plot(kind='pie', labels=["pol", "inf"])
plt.title("Conditions repartition for the dataset.")
plt.show()

# we just saw that the condition repartition is pretty equal for polite and
# informal reponses.

#################################
# Boxplot for visualize the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sns.catplot(x='condition', y="pitch",
            hue='subject', data=df_politeness, kind="violin", legend=False,
            aspect=2)
plt.title("Pitch by conditions for each subjects")
plt.legend(loc='best')
plt.tight_layout()

# we saw that male subjects have lower voices than female subjects
# furthemore, we have noticed that there are individual variation for each sex
# indeed, some females got higher values that others

#####################################################
# Modeling individual means with random intercepts
#####################################################

# the formula is pitch=int+politeness+gender+(1|subject)+\varepsilon

# idea for the different participant means across conditions
df_politeness.groupby('subject')[['pitch']].aggregate(lambda x: x.mean())

# estimation of the means for each participant including a random intercept
# for each subject
md1 = smf.mixedlm("pitch ~ 1", df_politeness, groups=df_politeness['subject'])
md1f = md1.fit()
print(md1f.summary())

# we can see that the mean pitch is 193.025.

#############################
# Including fixed effects
#############################

# recoding variables gender and condition
df_politeness["gender"].replace({'F': 1, 'M': -1}, inplace=True)
df_politeness["condition"].replace({'inf': 1, 'pol': -1}, inplace=True)

# include a model with condition and gender and a random intercept
# for each subject
md2 = smf.mixedlm("pitch ~ condition + gender", df_politeness,
                  groups=df_politeness['subject'])
md2f = md2.fit()
print(md2f.summary())

# we can see that the mean pitch is 192.883.
# we noticed that pitch is higher for informal than polite.
# we noticed that pitch is higher for females than males.
# the p-values are < 5% so the coefficients are significate
# we constate that the confidence interval of the covariable gender is large

###########################
# more model informations
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# deviance=−2∗log likelihood
# AIC=deviance+2∗(p+1)

dev_md2 = (-2)*md2f.llf
print("model's deviance : %.4f" % (dev_md2))
p = 4  # number of parameters = 3 (fixed) + 1 (random)
print("model's AIC : %.4f" % (dev_md2 + 2*(p+1)))
# total parameters = 4 + 1 for estimated residual

#####################
# Random Slopes
#####################

md3 = smf.mixedlm("pitch ~ condition + gender", df_politeness,
                  groups=df_politeness['subject'], re_formula='~condition')
md3f = md3.fit()
print(md3f.summary())

# anova of md2 vs md3
dev_md3 = (-2)*md3f.llf
AIC_md2 = dev_md2 + 2*(4+1)
AIC_md3 = dev_md3 + 2*(6+1)
# we see that the AIC of the model 2 is smaller than the AIC of the model 3
# so, the best model is the model 2

dev_diff = dev_md2 - dev_md3
pvalue = 1.0 - scipy.stats.chi2.cdf(dev_diff, 2)

print('Chi square =', np.round(dev_diff, 3), '(df=2)',
      'p=', np.round(pvalue, 6))
# adding random slopes for each subject takes up 2 more degrees of freedom
# with the p-value (equals 1), we can note that it doesn't improve model fit

#######################
# Testing signifiance
#######################

#####################
# Getting p-values
# ~~~~~~~~~~~~~~~~~~

md2b = md2.fit(reml=True)
md3b = md3.fit(reml=True)
print(md3b.summary())
# the p-values are <5% for both

#####################
# Model comparison
# ~~~~~~~~~~~~~~~~~~

# comparison with likelihood ratio tests
# the likelihood ratio test essentially tells us how much more likely the data
# is under a more complex model than under the simpler model
# D =-2* ln(likelihood for simple model) + 2*ln(likelihood for complex model)
# the distribution of D is approximately chi_2 with df2-df1 degrees of freedom

md4 = smf.mixedlm("pitch ~  gender", df_politeness,
                  groups=df_politeness['subject'])
md4f = md4.fit(reml=False)
print(md4f.summary())

md4b = smf.mixedlm("pitch ~ condition + gender", df_politeness,
                   groups=df_politeness['subject'])
md4bf = md4b.fit(reml=False)
print(md4bf.summary())

dev_md4b = (-2)*md4bf.llf  # deviance complex model
dev_md4 = (-2)*md4f.llf  # deviance simpler model
AIC_md4 = dev_md4 + 2*(3+1)
AIC_md4b = dev_md4b + 2*(4+1)

# we see that the AIC of the model 4b is smaller than the AIC of the model 4
# so, the best model is the model 4b

dev_diff = dev_md4 - dev_md4b
print("Stat. of the likehood ratio : %.4f " % (dev_diff))

# params md4bf : 3 fixed + 1 random
# params md4f : 2 fixed + 1 random
# donc df_diff vaut 1

pvalue = 1.0 - scipy.stats.chi2.cdf(dev_diff, 1)

print('Chi square =', np.round(dev_diff, 3), '(df=1)',
      'p=', np.round(pvalue, 6))

# we have compared 2 nested models
# one without condition and other with condition
# we conclude that inclusion of condition is good in our model because
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

# AIC for the model md5f
print("model's AIC : %.4f" % (dev1b + 2*(4+1)))
# total parameters = 4 + 1 for estimated residual

# AIC for the model md5b
print("model's AIC : %.4f" % (dev0b + 2*(3+1)))
# total parameters = 3 +1 for estimated residual

# the AIC of the model 5 is smaller, so the model md5 is better than other one


# difference of the AIC for each model
print("the difference of AIC is :  %.4f" %
      np.round((dev0b + 2*(3+1))-(dev1b + 2*(4+1)), 3))


md5c = smf.mixedlm('pitch ~ condition + gender',
                   groups=df_politeness['subject'], data=df_politeness)
md5cf = md5.fit(reml=False)
dev0c = (-2)*md5cf.llf  # deviance of the model md5c
AIC_c = dev0c + 2*(4+1)  # AIC of the model md5c
dev0b = (-2)*md5b.llf  # deviance of the model md5b
AIC_b = dev0b + 2*(3+1)  # AIC of the model md5b
# the best model is the model md5c because its AIC is minimal

# now we compare the BIC

# BIC formula : -2 * loglikehood + (number of params + 1) * ln(numbers of obs)
BIC_c = (-2)*(md5cf.llf) + 5 * np.log(83)
BIC_b = (-2)*(md5b.llf) + 4 * np.log(83)
# we have 83 observations because we erase the observation that the pitch
# correspond to a NA-value
# the best model is also the model md5c because its BIC is minimal

dev_diff = dev0b - dev0c
p_value = 1.0 - scipy.stats.chi2.cdf(dev_diff, 1)
print('Chi square =', np.round(dev_diff, 3), '(df=1)',
      'p=', np.round(p_value, 6))
# the inclusion of random intercepts for subjects is warranted
# indeed χ2(1) = 19.51, p =1e-5

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
# we can see that the scenario 4 get pitch values higher than the others

