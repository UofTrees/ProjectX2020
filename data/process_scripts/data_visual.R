trial_num = 2
# change working directory. Don't put the trial num in the string because we are pasting it
setwd(paste0("~/Desktop/ProjectX/datasets/trial", toString(trial_num)))

# Begin analysis
# Load required packages 
library(ggplot2) 
library(ggcorrplot)

data <- read.csv("toy.csv")

model <- lm(num_infect~CT+RH+CM,data=data)
summary(model)
f = summary(model)$fstatistic
p = pf(f[1], f[2], f[3], lower.tail=F)

par(mfrow=c(1,3))
boxplot(data$CT, main="CT")
boxplot(data$RH, main="RH")
boxplot(data$CM, main="CM")

corr = cor(data[, c("RH", "CM", "CT", "num_infect")])
ggcorrplot(corr)


