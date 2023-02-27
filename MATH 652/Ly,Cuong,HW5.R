# -------------
# Problem 7.1
# -------------
data <- matrix(c(10,5,7,19,11,8,15,9,3,25,7,13),ncol=2,byrow=FALSE)
colnames(data) <- c("Z1",'Y')

# produce betahat results directly via method of least squares
ones <- rep(1,nrow(data))
z <- as.matrix(cbind(ones,data[,1])) 
betahat <- solve(t(z) %*% z) %*% t(z) %*% data[,2]

yhat <- z %*% betahat
epsilonhat <- data[,2] - yhat
rss <- t(data[,2]) %*% data[,2] - t(data[,2]) %*% z %*% betahat

data <- data.frame(data)
data.lm <- lm(Y ~ Z1, data = data)
summary(data.lm)
# yhat = -2/3 + (19/15)*Z1

# --------------
# Problem 7.19
# --------------
battery <- read.table('/Users/taikhanghao/Desktop/spring 23/multi/T7-5.DAT')
colnames(battery) <- c("Z1","Z2",'Z3','Z4','Z5','Y')
ln_y <- log(battery[,6])
battery <- as.matrix(cbind(battery[,1:6],ln_y)) 

battery <- data.frame(battery)

# Use all the variables for regression
battery.lm12345 <- lm(ln_y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = battery)
summary(battery.lm12345)
# anova(battery.lm)
# Observe that Z3 has significance p-value --> Drop Z3

battery.lm1245 <- lm(ln_y ~ Z1 + Z2 + Z4 + Z5, data = battery)
summary(battery.lm1245)
# Observe that Z1 has significance p-value --> Drop Z1

battery.lm245 <- lm(ln_y ~ Z2 + Z4 + Z5, data = battery)
summary(battery.lm245)

# Observe that Z5 has significance p-value --> Drop Z5
battery.lm24 <- lm(ln_y ~ Z2 + Z4 , data = battery)
summary(battery.lm24)

# Thus the regression result is ln(Y) = B1*(Z2) + B2*(Z4) + e

# Part b

par(mfrow=c(2,2))
plot(battery.lm24,add.smooth=FALSE)
shapiro.test(battery.lm24$residuals)
# Normal 