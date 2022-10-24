fabric = read.table("fabric.txt", header=TRUE)

plot(faults~length, data=fabric)
plot(log(faults)~length, data=fabric)


#glm
?glm
glm_fit = glm(faults~length, data=fabric, family="poisson")
summary(glm_fit)
names(glm_fit)
methods(class=class(glm_fit))
glm_fit_fisher_info = vcov(glm_fit)

util1 <- function(x_eval_pt, fitted_beta, fitted_info){
    eval_exp = c(1, x_eval_pt)
    pt_est = t(eval_exp) %*% fitted_beta
    interval_length = qnorm(0.975) * sqrt(t(eval_exp) %*% fitted_info %*% eval_exp)
    lw_est = pt_est - interval_length
    up_est = pt_est + interval_length
    return(data.frame(pt=pt_est, lw=lw_est, up=up_est))
}

util1(500, coef(glm_fit), vcov(glm_fit))
util1(995, coef(glm_fit), vcov(glm_fit))


#quasi-likelihood glm
qglm_fit = glm(faults~length, data=fabric, family="quasipoisson")
summary(qglm_fit)

util1(500, coef(qglm_fit), vcov(qglm_fit))
util1(995, coef(qglm_fit), vcov(qglm_fit))
