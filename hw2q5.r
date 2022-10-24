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

fit2 <- glm(num ~ concentration, data=cerio, family="poisson")
summary(fit2)
fit2_dev = fit2$deviance
(1 - pchisq(fit2$deviance, fit2$df.residual)) #bad

fit3 <- glm(num ~ strain, data=cerio, family="poisson")
summary(fit3)
fit3_dev = fit3$deviance
(1 - pchisq(fit3$deviance, fit3$df.residual)) #bad

fit4 <- glm(num ~ 1, data=cerio, family="poisson")
summary(fit4)
fit4_dev = fit4$deviance
(1 - pchisq(fit4$deviance, fit4$df.residual)) #bad


# model selection
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

#choose fit1!


