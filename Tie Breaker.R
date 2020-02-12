library(lattice)
MyData <- read.csv(file="C:/Users/HP/Downloads/project/project regression/Player-1.csv", header=TRUE, sep=",")
MyData
MyData[, 1:6]
Regression1<- lm(won ~ Rank+height+BMI+firstserve+B365, data = MyData)
Regression1
b11 <- coef(Regression1)[5]
b11
summary(Regression1)
b12<-summary(Regression1)$r.squared
Regression2<- lm(won ~ Rank+height+BMI+firstserve, data = MyData)
Regression2
b21 <- coef(Regression2)[5]
b21
summary(Regression2)
b22<-summary(Regression2)$r.squared
d<-1
rmax<-(1.3*b12)
rmax
b1<-b11-d*(b21-b11)*(rmax-b12)/(b12-b22)
cat("The bias for male player 1 is",b1)
bmi = MyData[,4]
rank = MyData[,2]
won = MyData[,1]
bmi
rank
plot(bmi,rank)
xyplot(bmi ~ rank, group=won, data=MyData, 
       auto.key=list(space="right"), 
       jitter.x=TRUE, jitter.y=TRUE)

MyData <- read.csv(file="C:/Users/HP/Downloads/project/project regression/Player-2.csv", header=TRUE, sep=",")
MyData
MyData[, 1:6]
Regression1<- lm(won ~ Rank+height+BMI+firstserve+B365, data = MyData)
Regression1
b11 <- coef(Regression1)[5]
b11
summary(Regression1)
b12<-summary(Regression1)$r.squared
b12
Regression2<- lm(won ~ Rank+height+BMI+firstserve, data = MyData)
Regression2
b21 <- coef(Regression2)[5]
b21
summary(Regression2)
b22<-summary(Regression2)$r.squared
b22
d<-1
d
rmax<-(1.3*b12)
rmax
b2<-b11-d*(b21-b11)*(rmax-b12)/(b12-b22)
cat("the bias for male player 2 is",b2)
bmi = MyData[,4]
rank = MyData[,2]
won = MyData[,1]
bmi
rank
plot(bmi,rank)
xyplot(bmi ~ rank, group=won, data=MyData, 
       auto.key=list(space="right"), 
       jitter.x=TRUE, jitter.y=TRUE)
MyData <- read.csv(file="C:/Users/HP/Downloads/project/project regression/Player-1-f.csv", header=TRUE, sep=",")
MyData
MyData[, 1:6]
Regression1<- lm(won ~ Rank+height+BMI+firstserve+B365, data = MyData)
Regression1
b11 <- coef(Regression1)[5]
b11
summary(Regression1)
b12<-summary(Regression1)$r.squared
b12
Regression2<- lm(won ~ Rank+height+BMI+firstserve, data = MyData)
Regression2
b21 <- coef(Regression2)[5]
b21
summary(Regression2)
b22<-summary(Regression2)$r.squared
b22
d<-1
d
rmax<-(1.3*b12)
rmax
b3<-b11-d*(b21-b11)*(rmax-b12)/(b12-b22)
cat("the bias for female player 1 is",b3)
bmi = MyData[,4]
rank = MyData[,2]
won = MyData[,1]
bmi
rank
plot(bmi,rank)
xyplot(bmi ~ rank, group=won, data=MyData, 
       auto.key=list(space="right"), 
       jitter.x=TRUE, jitter.y=FALSE)
MyData <- read.csv(file="C:/Users/HP/Downloads/project/project regression/Player-2-f.csv", header=TRUE, sep=",")
MyData
MyData[, 1:6]
Regression1<- lm(won ~ Rank+height+BMI+firstserve+B365, data = MyData)
Regression1
b11 <- coef(Regression1)[5]
b11
summary(Regression1)
b12<-summary(Regression1)$r.squared
b12
Regression2<- lm(won ~ Rank+height+BMI+firstserve, data = MyData)
Regression2
b21 <- coef(Regression2)[5]
b21
summary(Regression2)
b22<-summary(Regression2)$r.squared
b22
d<-1
d
rmax<-(1.3*b12)
rmax
b4<-b11-d*(b21-b11)*(rmax-b12)/(b12-b22)
cat("the bias for female player 2 is",b4)
bmi = MyData[,4]
rank = MyData[,2]
won = MyData[,1]
bmi
rank
plot(bmi,rank)
xyplot(bmi ~ rank, group=won, data=MyData, 
       auto.key=list(space="right"), 
       jitter.x=TRUE, jitter.y=TRUE)


print("THE FINAL RESULTS FOR BIAS ADJUSTMENT TREATMENT ARE:")
cat("The bias for male player 1 is",b1)
cat("The bias for male player 2 is",b2)
cat("The bias for female player 1 is",b3)
cat("The bias for female player 1 is",b4)
