# Visualization of training.
# 
# Author: tombr
###############################################################################

require(ggplot2)
require(reshape)

logname = "logs/layer1_p4_f32_e1000.csv"
#logname = "logs/nusparse_l1_p4_f32_e500.csv"
#logname = my.file.browse();
training.mean = subset(read.csv(logname), select = c("F.mean", "c.mean", "b.mean", "h.mean", "vneg.mean", "error"))
training.mean = data.frame(id=0:(nrow(training.mean)-1), training.mean)
training.mean = melt(training.mean, id = "id")

names(training.mean)[names(training.mean)=="value"] <- "mean"
print(head(training.mean))

training.sd = subset(read.csv(logname), select = c("F.sd", "c.sd"))
training.sd = data.frame(id=0:(nrow(training.sd)-1), training.sd, b.sd = 0, h.sd = 0, vneg.sd = 0, error.sd = 0)
#training.sd = subset(read.csv(logname), select = c("F.sd"))
#training.sd = data.frame(id=0:(nrow(training.sd)-1), training.sd)
training.sd = melt(training.sd, id = "id")
names(training.sd)[names(training.sd)=="value"] <- "sd"
print(head(training.sd))

training = data.frame(training.mean, sd = training.sd$sd)
print(head(training))
myplot = ggplot(training, aes(x = id))
myplot = myplot + geom_ribbon(aes(ymin=mean-2*sd, ymax=mean+2*sd, fill = variable), alpha=0.5)
myplot = myplot + geom_line(aes(y=mean, colour = variable))
print(myplot)
