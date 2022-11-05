# MCMC
from random import seed, normalvariate
from math import exp, log
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from pyBayes import MCMC_Core

seed(20221105)



# data
x_m_y = [
    (1.6907, 59, 6), 
    (1.7242, 60, 13),
    (1.7552, 62, 18),
    (1.7842, 56, 28),
    (1.8113, 63, 52),
    (1.8369, 59, 53),
    (1.8610, 62, 61),
    (1.8839, 60, 60)
]



# all MCMC settings
def symmetric_proposal_placeholder(from_smpl, to_smpl):
    #for log_proposal
    return 0

def normal_proposal_sampler(from_smpl, proposal_sigma_vec):
    return [normalvariate(x, proposal_sigma_vec[i]) for i,x in enumerate(from_smpl)]
    

def cloglog_posterior_with_flat_prior(beta, data):
    log_post = 0
    for (x, m, y) in data:
        nu = beta[0] + beta[1]*x
        log_post += (y*log(1-exp(-exp(nu))) + (m-y)*(-exp(nu)))
    return log_post

def cloglog_posterior_with_normal_prior(beta, data, sigma1, sigma2):
    log_post = cloglog_posterior_with_flat_prior(beta, data)
    log_post += (-0.5*beta[0]**2/(sigma1**2) - 0.5*beta[1]**2/(sigma2**2))
    return log_post

def logit_posterior_with_flat_prior(beta, data):
    log_post = 0
    for (x, m, y) in data:
        nu = beta[0] + beta[1]*x
        log_post += (y*nu - m*log(1+exp(nu)))
    return log_post

def powered_logit_posterior_with_normal_prior(beta_lambda, data, sigma1, sigma2, sigma_lambda):
    # beta_lambda=[beta1, beta2, lambda]
    log_post = 0
    e_lam = exp(beta_lambda[2])
    for (x, m, y) in data:
        nu = beta_lambda[0] + beta_lambda[1]*x
        try:
            log_post += (y*e_lam*nu - y*e_lam*log(1+exp(nu)) + (m-y)*log(1-(exp(nu)/(1+exp(nu)))**e_lam))
        except ValueError:
            if (m-y) == 0:
                log_post += (y*e_lam*nu - y*e_lam*log(1+exp(nu)))
            else:
                raise ValueError("check it", m-y, beta_lambda[2], e_lam, 1-(exp(nu)/(1+exp(nu)))**e_lam)
    log_post += 0.5*(-beta_lambda[0]**2/sigma1**2 - beta_lambda[1]**2/sigma2**2 - beta_lambda[2]/sigma_lambda**2)
    return log_post



# other utilities
def pi_plot(posterior_samples, link_func): #depend on x_m_y
    x_grid = np.arange(1.6, 1.9, 0.005)
    lwr = []
    med = []
    upr = []
    avg = []
    for x in x_grid:
        pi_samples_at_x = [link_func(x, params) for params in posterior_samples]
        lwr.append(float(np.quantile(pi_samples_at_x, 0.025)))
        med.append(float(np.quantile(pi_samples_at_x, 0.5)))
        upr.append(float(np.quantile(pi_samples_at_x, 0.975)))
        avg.append(float(np.mean(pi_samples_at_x)))

    plt.ylim([0, 1.05])
    plt.plot(x_grid, avg, color="black")
    plt.plot(x_grid, med, color="red")
    plt.plot(x_grid, lwr, color="grey")
    plt.plot(x_grid, upr, color="grey")
    plt.plot([x[0] for x in x_m_y], [x[2]/x[1] for x in x_m_y], marker="o")
    plt.show()

def bayesian_residual_plot(posterior_samples, link_func): #depend on x_m_y
    x_grid = [x[0] for x in x_m_y]
    boxplot_frame = []
    for (x, m, y) in x_m_y:
        obs_pi = y/m
        res_samples_at_x = [link_func(x, params)-obs_pi for params in posterior_samples]
        boxplot_frame.append(res_samples_at_x)

    plt.boxplot(np.transpose(boxplot_frame), labels=x_grid)
    plt.axhline(0)
    plt.show()

def loss_L_measure(posterior_samples, link_func, k): #depend on x_m_y
    loss = 0
    for (x, m, y) in x_m_y:
        pi_samples_at_x = [link_func(x, params) for params in posterior_samples]
        rng = np.random.default_rng()
        y_predictive_samples_at_x = [rng.binomial(m, pi) for pi in pi_samples_at_x]
        predictive_var_at_x = np.var(y_predictive_samples_at_x)
        predictive_mean_at_x = np.mean(y_predictive_samples_at_x)
        loss += (predictive_var_at_x + (k/(k+1))*(y-predictive_mean_at_x)**2)
    return loss



##2a
def cloglog_link(x, params):
    # params: [beta1, beta2]
    nu = params[0]+params[1]*x
    return 1-exp(-exp(nu))

cloglog_flat_initial = [0,0]
cloglog_flat_inst = MCMC_Core.MCMC_MH(
                partial(cloglog_posterior_with_flat_prior, data=x_m_y),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[3, 3]),
                cloglog_flat_initial)
cloglog_flat_inst.generate_samples(200000, print_iter_cycle=20000)
cloglog_flat_diag = MCMC_Core.MCMC_Diag()
cloglog_flat_diag.set_mc_sample_from_MCMC_instance(cloglog_flat_inst)
cloglog_flat_diag.set_variable_names(["beta1", "beta2"])
cloglog_flat_diag.burnin(20000)
cloglog_flat_diag.thinning(40)
cloglog_flat_diag.show_traceplot((1,2))
cloglog_flat_diag.show_acf(30, (1,2))
cloglog_flat_diag.show_hist((1,2))

cloglog_flat_LD50_samples = [[(log(-log(0.5))-x[0])/x[1]] for x in cloglog_flat_diag.MC_sample]
cloglog_flat_LD50_diag = MCMC_Core.MCMC_Diag()
cloglog_flat_LD50_diag.set_mc_samples_from_list(cloglog_flat_LD50_samples)
cloglog_flat_LD50_diag.set_variable_names(["LD50"])
cloglog_flat_LD50_diag.show_hist((1,1))

pi_plot(cloglog_flat_diag.MC_sample, cloglog_link)
bayesian_residual_plot(cloglog_flat_diag.MC_sample, cloglog_link)
print("cloglog_flat L: ", loss_L_measure(cloglog_flat_diag.MC_sample, cloglog_link, 10)) #105.81742676883584

# ===

cloglog_normal_initial = [0,0]
cloglog_normal_initial_inst = MCMC_Core.MCMC_MH(
                partial(cloglog_posterior_with_normal_prior, data=x_m_y, sigma1=10, sigma2=10),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[3, 3]),
                cloglog_normal_initial)
cloglog_normal_initial_inst.generate_samples(200000, print_iter_cycle=20000)
cloglog_normal_diag = MCMC_Core.MCMC_Diag()
cloglog_normal_diag.set_mc_sample_from_MCMC_instance(cloglog_normal_initial_inst)
cloglog_normal_diag.set_variable_names(["beta1", "beta2"])
cloglog_normal_diag.burnin(20000)
cloglog_normal_diag.thinning(40)
cloglog_normal_diag.show_traceplot((1,2))
cloglog_normal_diag.show_acf(30, (1,2))
cloglog_normal_diag.show_hist((1,2))

cloglog_normal_LD50_samples = [[(log(-log(0.5))-x[0])/x[1]] for x in cloglog_normal_diag.MC_sample]
cloglog_normal_LD50_diag = MCMC_Core.MCMC_Diag()
cloglog_normal_LD50_diag.set_mc_samples_from_list(cloglog_normal_LD50_samples)
cloglog_normal_LD50_diag.set_variable_names(["LD50"])
cloglog_normal_LD50_diag.show_hist((1,1))

pi_plot(cloglog_normal_diag.MC_sample, cloglog_link)
bayesian_residual_plot(cloglog_normal_diag.MC_sample, cloglog_link)
print("cloglog_normal L: ", loss_L_measure(cloglog_normal_diag.MC_sample, cloglog_link, 10)) #129.2217430392814



##2b
def logit_link(x, params):
    # params: [beta1, beta2]
    nu = params[0]+params[1]*x
    return exp((nu))/(1+exp(nu))

logistic_flat_initial = [0,0]
logistic_flat_initial_inst = MCMC_Core.MCMC_MH(
                partial(logit_posterior_with_flat_prior, data=x_m_y),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[3, 3]),
                logistic_flat_initial)
logistic_flat_initial_inst.generate_samples(200000, print_iter_cycle=20000)
logistic_flat_diag = MCMC_Core.MCMC_Diag()
logistic_flat_diag.set_mc_sample_from_MCMC_instance(logistic_flat_initial_inst)
logistic_flat_diag.set_variable_names(["beta1", "beta2"])
logistic_flat_diag.burnin(20000)
logistic_flat_diag.thinning(40)
logistic_flat_diag.show_traceplot((1,2))
logistic_flat_diag.show_acf(30, (1,2))
logistic_flat_diag.show_hist((1,2))


logistic_flat_LD50_samples = [[-x[0]/x[1]] for x in logistic_flat_diag.MC_sample]
logistic_flat_LD50_diag = MCMC_Core.MCMC_Diag()
logistic_flat_LD50_diag.set_mc_samples_from_list(logistic_flat_LD50_samples)
logistic_flat_LD50_diag.set_variable_names(["LD50"])
logistic_flat_LD50_diag.show_hist((1,1))

pi_plot(logistic_flat_diag.MC_sample, logit_link)
bayesian_residual_plot(logistic_flat_diag.MC_sample, logit_link)
print("logistic_flat L: ", loss_L_measure(logistic_flat_diag.MC_sample, logit_link, 10)) #144.67401288130463



##2c
def pw_logit_link(x, params):
    # params: [beta1, beta2, lambda, alpha]
    nu = params[0]+params[1]*x
    return exp(params[3]*(nu))/(1+exp(nu))**params[3]

pw_logistic_normal_initial = [0,0,0]
pw_logistic_normal_initial_inst = MCMC_Core.MCMC_MH(
                partial(powered_logit_posterior_with_normal_prior, data=x_m_y, sigma1=40, sigma2=20, sigma_lambda=0.2),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[3, 3, 0.4]),
                pw_logistic_normal_initial)
pw_logistic_normal_initial_inst.generate_samples(200000, print_iter_cycle=20000)

pw_samples_appended = [smpl + [exp(smpl[2])] for smpl in pw_logistic_normal_initial_inst.MC_sample]

pw_logistic_normal_diag = MCMC_Core.MCMC_Diag()
pw_logistic_normal_diag.set_mc_samples_from_list(pw_samples_appended)
pw_logistic_normal_diag.set_variable_names(["beta1", "beta2", "lambda", "alpha"])
pw_logistic_normal_diag.burnin(20000)
pw_logistic_normal_diag.thinning(40)
pw_logistic_normal_diag.show_traceplot((2,2))
pw_logistic_normal_diag.show_acf(30, (2,2))
pw_logistic_normal_diag.show_hist((2,2))


pw_logistic_normal_LD50_samples = [[(log(0.5**(1/x[3])/(1-0.5**(1/x[3])))-x[0])/x[1]] for x in pw_logistic_normal_diag.MC_sample]
pw_logistic_normal_LD50_diag = MCMC_Core.MCMC_Diag()
pw_logistic_normal_LD50_diag.set_mc_samples_from_list(pw_logistic_normal_LD50_samples)
pw_logistic_normal_LD50_diag.set_variable_names(["LD50"])
pw_logistic_normal_LD50_diag.show_hist((1,1))

pi_plot(pw_logistic_normal_diag.MC_sample, pw_logit_link)
bayesian_residual_plot(pw_logistic_normal_diag.MC_sample, pw_logit_link)
print("pw_logistic_normal L: ", loss_L_measure(pw_logistic_normal_diag.MC_sample, pw_logit_link, 10)) # 144.06169963213338