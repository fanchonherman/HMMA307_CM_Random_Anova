from download import download
from statsmodels.formula.api import ols
from plotnine import *
from plotnine.data import mpg
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.graphics.factorplots import interaction_plot
import numpy as np
import seaborn as sns
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
################################
# Download datasets
#################################
#A group of 10 “rural” and 10 “urban” raters rated 4 different cheese types(A,B,C,D). Every rater got to eat two
# samples from the same cheese type in random order. Hence, we have a total of 160 observations.

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

cheese = "http://stat.ethz.ch/~meier/teaching/data/cheese.dat"
path_cheese = os.path.join(results_dir, "cheese.csv")
download(cheese, path_cheese, replace=False)
df_cheese = pd.read_csv(path_cheese, sep=" ", header=0)
############## make rater as a factor########
df_cheese.rater=pd.Categorical(df_cheese.rater)

## the factor rater has only 10 Levels. We have to be careful here and should not forget that rater is actually nested in background.
df_cheese.info()
print(df_cheese)


######################### data visualisation #############################

#We can easily visualize this data with an interaction plot.
#We use the package ggplot2 to get a more appealing plot compared to the function interaction by rater.

ggplot(df_cheese, aes(x = df_cheese.cheese, y = df_cheese.y, group = interaction(df_cheese.background, df_cheese.rater),
                   color = df_cheese.rater)) + stat_summary(df_cheese.y = mean, geom = "line")

######We have main effects and the interaction for the fixed effects of background and cheese type: background * cheese
#####a random effect per rater (nested in background): (1 | rater:background)
#####a random effect per cheese type and rater (nested in background): (1 | rater:background:cheese).
#####We always write rater:background to get a unique rater ID. An alternative would be to define another factor in the data set which enumerates the raters from 1 to 20.

mixmod = smf.mixedlm("y ~ (background * cheese) + (1 - rater*background)+ (1 - rater*background*cheese) ", data = df_cheese, groups="rater")
mode = mixmod.fit()
mode.summary()

###########  ANOVA  ##############
#We get an ANOVA table with p-values for the fixed effects, after fiting an ols model.

model = ols('y ~ C(background, rater, cheese)', data=df_cheese).fit(
anova_model  =  sm.stats.anova_lm ( model )   

###We see that the interaction is not significant but there is a significant effect of background and cheese type. This is what we basically already observed in the interaction plot. There, the profiles were quite parallel, but raters with urban background rated higher on average than those with rural background.
#In addition, there was a clear difference between different cheese types.


############ confidence intervals ######
#Hence, if we want to get a more precise view about these population average effects, we need to increase the number of raters, which seems pretty natural.
#Approximate confidence intervals for the individual coefficients of the fixed effects and the variance components.
model.conf_int(alpha=0.025)