log_dose <- c(1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839)
beetles <- c(59, 60, 62, 56, 63, 59, 62, 60) #m
killed <- c(6, 13, 18, 28, 52, 53, 61, 60) #y

#1a
# ?glm
# ?binomial
logit_fit = glm(cbind(killed, beetles-killed)~log_dose, family=binomial(link="logit"))
summary(logit_fit)
probit_fit = glm(cbind(killed, beetles-killed)~log_dose, family=binomial(link="probit"))
summary(probit_fit)
cloglog_fit = glm(cbind(killed, beetles-killed)~log_dose, family=binomial(link="cloglog"))
summary(cloglog_fit)

res_logit = residuals(logit_fit, "deviance")
res_probit = residuals(probit_fit, "deviance")
res_cloglog = residuals(cloglog_fit, "deviance")

plot(log_dose, res_logit, col="orange", xlim=c(1.65, 1.9), cex=3)
points(log_dose, res_probit, col="red", cex=3)
points(log_dose, res_cloglog, col="blue", cex=3)
abline(h=0)

inv_logit <- function(x, beta_0, beta_1){
    return(exp(beta_0 + beta_1*x) / (1 + exp(beta_0 + beta_1*x)))
}
inv_probit <- function(x, beta_0, beta_1){
    return(pnorm(beta_0 + beta_1*x))
}
inv_cloglog <- function(x, beta_0, beta_1){
    return(1-exp(-exp(beta_0 + beta_1*x)))
}
# grid = seq(1.65, 1.9, 0.005)
plot(log_dose, killed/beetles, xlim=c(1.65, 1.9), cex=3)
curve(inv_logit(x, logit_fit$coefficients[1], logit_fit$coefficients[2]), add=T, col="orange")
curve(inv_probit(x, probit_fit$coefficients[1], probit_fit$coefficients[2]), add=T, col="red")
curve(inv_cloglog(x, cloglog_fit$coefficients[1], cloglog_fit$coefficients[2]), add=T, col="blue")


#1c: alpha-power logit link
pll_beta0 = -113.625
pll_beta1 = 62.5
pll_alpha = 0.279

pll_inv <- function(x, beta_0, beta_1, alpha){
    nu = beta_0 + beta_1*x
    return(exp(alpha*nu)/((1+exp(nu))^alpha))
}
(pll_fit = pll_inv(log_dose, pll_beta0, pll_beta1, pll_alpha))
plot(log_dose, killed/beetles, xlim=c(1.65, 1.9), cex=3)
curve(inv_logit(x, logit_fit$coefficients[1], logit_fit$coefficients[2]), add=T, col="orange")
curve(inv_probit(x, probit_fit$coefficients[1], probit_fit$coefficients[2]), add=T, col="red")
curve(inv_cloglog(x, cloglog_fit$coefficients[1], cloglog_fit$coefficients[2]), add=T, col="blue")
curve(pll_inv(x, pll_beta0, pll_beta1, pll_alpha), add=T, col="green")

pll_dev_res <- function(m, y, fitted_pi){
    red_theta = log(fitted_pi/(1-fitted_pi))
    full_theta = log(y/(m-y))
    d = 2*(y*(full_theta - red_theta) + m*log((1+exp(red_theta))/(1+exp(full_theta))))
    sign = ifelse((y/m - fitted_pi)>0, +1, -1)
    return(sign*sqrt(d))
}
#debug
# pll_dev_res(beetles, killed, logit_fit$fitted.values)
# (res_logit = residuals(logit_fit, "deviance"))

#pll dev.resid
pll_dev_res(beetles, killed, pll_fit) #cause m=y for last point, it gives NaN


# 1d. AIC / BIC
pll_log_lik = 0
for(i in 1:8){
    pll_log_lik = pll_log_lik + dbinom(killed[i],beetles[i],pll_fit[i], log=TRUE)
}
(pll_log_lik)

pll_num_param = 3
(pll_AIC = -2*pll_log_lik + 2*pll_num_param)
(pll_BIC = -2*pll_log_lik + pll_num_param * log(8))

AIC(logit_fit)
BIC(logit_fit)
AIC(probit_fit)
BIC(probit_fit)
AIC(cloglog_fit)
BIC(cloglog_fit)
