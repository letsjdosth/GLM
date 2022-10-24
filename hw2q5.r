
cerio = read.table("ceriodaphnia.txt")
colnames(cerio) <- c("num", "concentration", "strain")
head(cerio)
cerio$strain <- factor(cerio$strain)

plot(log(num) ~ concentration, data=cerio)
plot(log(num) ~ strain, data=cerio)

#residual deviance: deviance for the model having only the intercept term
#GOF: H0: smaller, H1: full (less D == good fit)

fit1 <- glm(num ~ concentration + strain, data=cerio, family="poisson")
summary(fit1)
fit1_dev = fit1$deviance
(1 - pchisq(fit1$deviance, fit1$df.residual)) #good
AIC(fit1) #415.9508
BIC(fit1) #422.6963

fit2 <- glm(num ~ concentration, data=cerio, family="poisson")
summary(fit2)
fit2_dev = fit2$deviance
(1 - pchisq(fit2$deviance, fit2$df.residual)) #bad
AIC(fit2) #446.5694
BIC(fit2) #451.0664

fit3 <- glm(num ~ strain, data=cerio, family="poisson")
summary(fit3)
fit3_dev = fit3$deviance
(1 - pchisq(fit3$deviance, fit3$df.residual)) #bad
AIC(fit3) #1654.337
BIC(fit3) #1658.834

fit4 <- glm(num ~ 1, data=cerio, family="poisson")
summary(fit4)
fit4_dev = fit4$deviance
(1 - pchisq(fit4$deviance, fit4$df.residual)) #bad
AIC(fit4) #1684.955
BIC(fit4) #1678.204

# model selection via deviance
# H0: smaller, H1: larger
# chi2 statistic = D(H0)- D(H1)
(dev21 = fit2_dev - fit1_dev)
(df21 = fit2$df.residual - fit1$df.residual)
(1 - pchisq(dev21, df21)) #near 0. reject H0. The larger one is better.

(dev31 = fit3_dev - fit1_dev)
(df31 = fit3$df.residual - fit1$df.residual)
(1 - pchisq(dev31, df31)) #0. reject H0. The larger one is better.

(dev42 = fit4_dev - fit2_dev)
(df42 = fit4$df.residual - fit2$df.residual)
(1 - pchisq(dev42, df42)) #0. reject H0. the larger one is better

(dev43 = fit4_dev - fit3_dev)
(df43 = fit4$df.residual - fit3$df.residual)
(1 - pchisq(dev43, df43)) #near 0. reject H0. the larger one is better

(dev41 = fit4_dev - fit1_dev)
(df41 = fit4$df.residual - fit1$df.residual)
(1 - pchisq(dev41, df41)) #0. reject H0. the larger one is better

#residuals
?residuals.glm
# residuals(fit1, "pearson")
# residuals(fit1, "deviance")

set.seed(20221024)

#simulate models - fit1
sim1_data = rep(0, length(cerio$num))
for(i in 1:length(cerio$num)){
    nu = 0
    if(cerio$strain[i]==0){
        nu = t(fit1$coefficients) %*% c(1, cerio$concentration[i], 0)
    } else {
        nu = t(fit1$coefficients) %*% c(1, cerio$concentration[i], 1)
    }
    mu = exp(nu)
    sim1_data[i] = rpois(1, mu)
}
glm_sim1_fit = glm(sim1_data~cerio$concentration + cerio$strain, family="poisson")
par(mfrow=c(1,2))
qqplot(residuals(fit1, "pearson"), residuals(glm_sim1_fit, "pearson"), main="pearson")
qqplot(residuals(fit1, "deviance"), residuals(glm_sim1_fit, "deviance"), main="deviance")

#simulate models - fit2
sim2_data = rep(0, length(cerio$num))
for(i in 1:length(cerio$num)){
    nu = t(fit2$coefficients) %*% c(1, cerio$concentration[i])
    mu = exp(nu)
    sim2_data[i] = rpois(1, mu)
}
glm_sim2_fit = glm(sim2_data~cerio$concentration, family="poisson")
par(mfrow=c(1,2))
qqplot(residuals(fit2, "pearson"), residuals(glm_sim2_fit, "pearson"), main="pearson")
qqplot(residuals(fit2, "deviance"), residuals(glm_sim2_fit, "deviance"), main="deviance")


#simulate models - fit3
sim3_data = rep(0, length(cerio$num))
for(i in 1:length(cerio$num)){
    nu = 0
    if(cerio$strain[i]==0){
        nu = t(fit3$coefficients) %*% c(1, 0)
    } else {
        nu = t(fit3$coefficients) %*% c(1, 1)
    }
    mu = exp(nu)
    sim3_data[i] = rpois(1, mu)
}
glm_sim3_fit = glm(sim3_data~cerio$strain, family="poisson")
par(mfrow=c(1,2))
qqplot(residuals(fit3, "pearson"), residuals(glm_sim3_fit, "pearson"), main="pearson")
qqplot(residuals(fit3, "deviance"), residuals(glm_sim3_fit, "deviance"), main="deviance")


#simulate models - fit4
sim4_data = rep(0, length(cerio$num))
for(i in 1:length(cerio$num)){
    nu = fit4$coefficients
    mu = exp(nu)
    sim4_data[i] = rpois(1, mu)
}
glm_sim4_fit = glm(sim4_data~1, family="poisson")
par(mfrow=c(1,2))
qqplot(residuals(fit4, "pearson"), residuals(glm_sim4_fit, "pearson"), main="pearson")
qqplot(residuals(fit4, "deviance"), residuals(glm_sim4_fit, "deviance"), main="deviance")


#choose fit1!
par(mfrow=c(1,3))
plot(log(num) ~ concentration, data=cerio[cerio["strain"]==0,], main="strain==0")
abline(fit1$coefficients[1], fit1$coefficients[2])
plot(log(num) ~ concentration, data=cerio[cerio["strain"]==1,], col="red", main="strain==1")
abline(fit1$coefficients[1]+fit1$coefficients[3], fit1$coefficients[2], col="red")

plot(log(num) ~ concentration, col=strain, data=cerio, main="all")
abline(fit1$coefficients[1], fit1$coefficients[2])
abline(fit1$coefficients[1]+fit1$coefficients[3], fit1$coefficients[2], col="red")
