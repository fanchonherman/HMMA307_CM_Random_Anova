"""
@authors: Fanchon Herman, Cassandre Lepercque and ..
In short: Usage of random Anova on two datasets.
"""

###################
# Packages needded
###################

##################
# Download datasets
##################

cheese <- read.table("http://stat.ethz.ch/~meier/teaching/data/cheese.dat", header = TRUE)
cheese[, "rater"] <- factor(cheese[, "rater"])
str(cheese)

