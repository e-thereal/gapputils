# Visualization of training.
# 
# Author: tombr
###############################################################################

require(ggplot2)
require(reshape)

#stats = subset(read.csv("training_nosparse.csv"), select = c("F.mean", "F.sd", "b", "c.mean", "c.sd"))
#stats2 = data.frame(id=0:(nrow(stats)-1), stats)
#myplot = ggplot(stats2, aes(x = id))
#myplot = myplot + geom_ribbon(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd))
#myplot = myplot + geom_line(aes(y=F.mean))
#print(myplot)

#stats = subset(read.csv("training.csv"), select = c("b", "binc", "c.mean", "cinc.mean", "F.mean", "Finc.mean"))
#stats = subset(read.csv("training_nosparse.csv"), select = c("b", "c.mean", "F.mean", "h.mean", "vneg.mean"))
#stats2 = melt(data.frame(id=0:(nrow(stats)-1), stats), id="id")
#head(stats2)
#print(qplot(id, (log(abs(value)+0.001) - log(0.001)) * sign(value), data = stats2, colour = variable, geom="line"))
#print(qplot(id, value, data = stats2, colour = variable, geom="line"))

#logname = "logs/layer1_f16_e200.csv"
logname = "logs/layer4_p2_f64_e1000.csv"
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
