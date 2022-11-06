# HW3Q3

from random import seed, normalvariate, gammavariate
from math import exp, log, lgamma
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgauss

from pyBayes import MCMC_Core

seed(20221106)


# Data set for homework 3, problem 3
# response y: machine tool failure time (minutes)
# covariate x: cutting speed (fpm)

failure_time = (70,29,60,28,64,32,44,24,35,31,38,35,52,23,40,28,46,33,46,27,37,34,41,28)
cutting_speed = (340,570,340,570,340,570,340,570,440,440,440,440,305,635,440,440,440,440,305,635,440,440,440,440)

x_y = [(x,y) for x,y in zip(cutting_speed, failure_time)]
# print(x_y)



# all MCMC settings
def symmetric_proposal_placeholder(from_smpl, to_smpl):
    #for log_proposal
    return 0

def normal_proposal_sampler(from_smpl, proposal_sigma_vec):
    return [normalvariate(x, proposal_sigma_vec[i]) for i,x in enumerate(from_smpl)]

def gamma_loglink_posterior(beta_lambda, data, sigma1, sigma2, sigma_lambda):
    beta1 = beta_lambda[0]
    beta2 = beta_lambda[1]
    lam = beta_lambda[2]
    v = exp(lam)

    log_post = 0
    for (x, y) in data:
        nu = beta1 + beta2*x
        try:
            log_post += (-lgamma(v) + v*log(v) -v*nu + (v-1)*log(y) - v*y/exp(nu))
        except ZeroDivisionError:
            raise ZeroDivisionError(beta1, beta2, x, nu, exp(nu))

    log_post += (-0.5*beta1**2/sigma1**2 - 0.5*beta2**2/sigma2**2 -0.5*lam**2/sigma_lambda**2)
    return log_post

def inv_gaussian_loglink_posterior(beta_lambda, data, sigma1, sigma2, sigma_lambda):
    beta1 = beta_lambda[0]
    beta2 = beta_lambda[1]
    lam = beta_lambda[2]
    phi = exp(lam)

    log_post = 0
    for (x, y) in data:
        nu = beta1 + beta2*x
        log_post += (-0.5*log(phi) - 1.5*log(y) - (y - exp(nu))**2 / (2*phi*y*exp(2*nu)))
    log_post += (-0.5*beta1**2/sigma1**2 - 0.5*beta2**2/sigma2**2 - 0.5*lam**2/sigma_lambda**2)
    return log_post

# utility
def mu_plot(posterior_samples, link_func): #depend on x_y
    x_grid = np.arange(300, 650, 1)
    lwr = []
    med = []
    upr = []
    avg = []
    for x in x_grid:
        mu_samples_at_x = [link_func(x, params) for params in posterior_samples]
        lwr.append(float(np.quantile(mu_samples_at_x, 0.025)))
        med.append(float(np.quantile(mu_samples_at_x, 0.5)))
        upr.append(float(np.quantile(mu_samples_at_x, 0.975)))
        avg.append(float(np.mean(mu_samples_at_x)))

    plt.plot(x_grid, avg, color="black")
    plt.plot(x_grid, med, color="red")
    plt.plot(x_grid, lwr, color="grey")
    plt.plot(x_grid, upr, color="grey")
    plt.scatter([x[0] for x in x_y], [x[1] for x in x_y], marker="o")
    plt.show()

def bayesian_residual_plot(posterior_samples, link_func): #depend on x_y
    x_grid = [x[0] for x in x_y]
    boxplot_frame = []
    for (x, y) in x_y:
        res_samples_at_x = [link_func(x, params)-y for params in posterior_samples]
        boxplot_frame.append(res_samples_at_x)

    plt.boxplot(np.transpose(boxplot_frame), labels=x_grid)
    plt.axhline(0)
    plt.show()

def loss_L_measure(posterior_samples, predictive_generator, k): #depend on x_y
    loss = 0
    for (x, y) in x_y:
        y_predictive_samples_at_x = [predictive_generator(x, param) for param in posterior_samples]
        predictive_var_at_x = np.var(y_predictive_samples_at_x)
        predictive_mean_at_x = np.mean(y_predictive_samples_at_x)
        loss += (predictive_var_at_x + (k/(k+1))*(y-predictive_mean_at_x)**2)
    return loss

def gamma_predictive_generator(x, param):
    #param: [beta1, beta2, lambda]
    shape = exp(param[2]) #v
    rate = shape/exp(param[0]+param[1]*x)
    return gammavariate(shape, 1/rate)

def inv_gaussian_predictive_generator(x, param): #need to check. is it right?
    #param: [beta1, beta2, lambda]
    mu = exp(param[0]+param[1]*x)
    phi = exp(param[2])
    return invgauss.rvs(mu=mu*phi, scale=1/phi)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html

def log_link(x, params):
    return exp(params[0]+params[1]*x)


gamma_log_initial = [0,0,0]
gamma_log_inst = MCMC_Core.MCMC_MH(
                partial(gamma_loglink_posterior, data=x_y, sigma1=50, sigma2=50, sigma_lambda=50),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.5, 0.005, 0.2]),
                gamma_log_initial)
gamma_log_inst.generate_samples(200000, print_iter_cycle=20000)

gamma_log_samples_appended = [smpl + [exp(smpl[2])] for smpl in gamma_log_inst.MC_sample]

gamma_log_diag = MCMC_Core.MCMC_Diag()
gamma_log_diag.set_mc_samples_from_list(gamma_log_samples_appended)
gamma_log_diag.set_variable_names(["beta1", "beta2", "log(v)", "v"])
gamma_log_diag.burnin(20000)
gamma_log_diag.thinning(50)
gamma_log_diag.show_traceplot((2,2))
gamma_log_diag.show_acf(30, (2,2))
gamma_log_diag.show_hist((2,2))


mu_plot(gamma_log_diag.MC_sample, log_link)
bayesian_residual_plot(gamma_log_diag.MC_sample, log_link)
print(loss_L_measure(gamma_log_diag.MC_sample, gamma_predictive_generator, 2)) 
print(loss_L_measure(gamma_log_diag.MC_sample, gamma_predictive_generator, 10)) 
#2236.2657701169755 at k=2
#2548.666959908084 at k=10

# ===

inv_gaussian_log_initial = [0,0,0]
inv_gaussian_log_inst = MCMC_Core.MCMC_MH(
                partial(inv_gaussian_loglink_posterior, data=x_y, sigma1=50, sigma2=50, sigma_lambda=50),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.5, 0.005, 0.3]),
                inv_gaussian_log_initial)
inv_gaussian_log_inst.generate_samples(200000, print_iter_cycle=20000)

inv_gaussian_log_samples_appended = [smpl + [exp(smpl[2])] for smpl in inv_gaussian_log_inst.MC_sample]

inv_gaussian_log_diag = MCMC_Core.MCMC_Diag()
inv_gaussian_log_diag.set_mc_samples_from_list(inv_gaussian_log_samples_appended)
inv_gaussian_log_diag.set_variable_names(["beta1", "beta2", "log(phi)", "phi"])
inv_gaussian_log_diag.burnin(20000)
inv_gaussian_log_diag.thinning(50)
inv_gaussian_log_diag.show_traceplot((2,2))
inv_gaussian_log_diag.show_acf(30, (2,2))
inv_gaussian_log_diag.show_hist((2,2))

mu_plot(inv_gaussian_log_diag.MC_sample, log_link)
bayesian_residual_plot(inv_gaussian_log_diag.MC_sample, log_link)
print(loss_L_measure(inv_gaussian_log_diag.MC_sample, inv_gaussian_predictive_generator, 2))
print(loss_L_measure(inv_gaussian_log_diag.MC_sample, inv_gaussian_predictive_generator, 10))
#2279.2581621040554 at k=2
#2586.9292392851125 at k=10
