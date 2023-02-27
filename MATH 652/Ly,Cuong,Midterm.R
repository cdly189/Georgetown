# ------------
# Question 4
# ------------
# Part a
x <- matrix(c(6,10,8,9,6,3),ncol=2)
dim(x)
n <- nrow(x)
p <- ncol(x)
mu0 <- c(9,5)
xbar <- colMeans(x)
xbar.mat <- matrix(rep(xbar,3),ncol=2,byrow=TRUE)
s <- (1/2)*t(x-xbar.mat)%*%(x-xbar.mat)
tsqstat <- n*t(xbar-mu0)%*%solve(s)%*%(xbar-mu0)
pval <- 1-pf(tsqstat/4,p,n-p)
pval2 <- pf(tsqstat/4,p,n-p,lower.tail=FALSE) # Fail to reject the null hypothesis

# Part b
micro <- matrix(c(6,10,8,9,6,3),ncol=2)
# micro <- cbind(closed,open)^(.25)
colnames(micro) <- c("closed","open")
micro.xbar <- colMeans(micro)
micro.xbarmat <- matrix(rep(micro.xbar,nrow(micro)),ncol=ncol(micro),byrow=TRUE)
#micro.s <- (1/(nrow(micro)-1)) * t(micro-micro.xbarmat) %*% (micro-micro.xbarmat)
micro.s <- (1/(nrow(micro)-1)) * t(as.matrix(micro-micro.xbarmat)) %*% as.matrix(micro-micro.xbarmat)
eigen(micro.s)
alpha <- 0.05 #let's suppose a 95% confidence region
fval <- qf(1-alpha,ncol(micro),nrow(micro)-ncol(micro))
sqrt(eigen(micro.s)$values[1])*sqrt(ncol(micro)*(nrow(micro)-1)*fval/(nrow(micro)*(nrow(micro)-ncol(micro))))
sqrt(eigen(micro.s)$values[2])*sqrt(ncol(micro)*(nrow(micro)-1)*fval/(nrow(micro)*(nrow(micro)-ncol(micro))))

#Slide 37:
n <- nrow(micro)
p <- ncol(micro)
t2low.micro <- micro.xbar - sqrt((p*(n-1)/(n-p))*fval)*sqrt(diag(micro.s)/n)
t2high.micro <- micro.xbar + sqrt((p*(n-1)/(n-p))*fval)*sqrt(diag(micro.s)/n)
t2.micro <- cbind(t2low.micro,t2high.micro)

# ------------
# Question 5
# ------------
college <- read.table('/Users/taikhanghao/Desktop/spring 23/multi/colleges.txt',header = TRUE)
library(MVN)
mvn(college[,3:8],mvnTest = 'mardia',univariateTest = "AD")
# There are 3 variables that normal are SAT, Acceptance and Top10Pct
# 3 variables that not normal are DoolarsPerStudent, PctPhD, GradPct

# Remove outliers in DollarsPerStudent using IQR method
dollarperstudent <- college[,5]
dollarperstudent <- 1/ dollarperstudent
qqnorm(dollarperstudent)
qqline(dollarperstudent)
shapiro.test(dollarperstudent)
# Normal at alpha = 0.01

PctPhD <- college[,7]
PctPhD <- PctPhD^6
qqnorm(PctPhD)
qqline(PctPhD)
shapiro.test(PctPhD)
# Normal at alpha = 0.01


gradPct <- college[,8]
gradPct <- gradPct^2
qqnorm(PctPhD)
qqline(PctPhD)
shapiro.test(gradPct^2)


# Part b
college[,5] <- dollarperstudent
college[,7] <- PctPhD
college[,8] <- gradPct
y <- cbind(college[,3:8])
college.manova <- manova(as.matrix(y) ~ School_Type, data=college)
summary.aov(college.manova)           # univariate ANOVA tables
summary(college.manova, test="Wilks") # ANOVA table of Wilks' lambda
# Since p-value is < 0.05, school types differ across all six dependent variables simultaneously

# Part c 
SAT <- aov( SAT ~ School_Type, data = college)
summary(SAT)
# Fail to reject the null hypothesis, SAT is the same across school type

Acceptance <- aov( Acceptance ~ School_Type, data = college)
summary(Acceptance)
# Fail to reject the null hypothesis, Acceptance is the same across school type

DollarPerStudent <- aov( DollarsPerStudent ~ School_Type, data = college)
summary(DollarPerStudent)
# Reject the null hypothesis, Dollars Per Student is significantly different across school type

Top10 <- aov( Top10Pct ~ School_Type, data = college)
summary(Top10)
# Reject the null hypothesis,  Top 10 Percent is significantly different across school type

pctPhD <- aov( PctPhD ~ School_Type, data = college)
summary(pctPhD)
# Reject the null hypothesis,  Pct PhD is significantly different across school type

GradPct <- aov( GradPct ~ School_Type, data = college)
summary(GradPct)
# Fail to reject the null hypothesis, GradPct is the same across school type

