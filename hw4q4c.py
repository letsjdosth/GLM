from random import seed, normalvariate
from math import exp, log
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from pyBayes import MCMC_Core

seed(20221120)

#data
concentration = np.array([0, 62.5, 125, 250, 500])
response_dead = np.array([15, 17, 22, 38, 144])
response_malformation = np.array([1, 0, 7, 59, 132])
response_normal = np.array([281, 225, 283, 202, 9])
num_subject = response_dead + response_malformation + response_normal

data_L1 = [(x, m, y1) for x,y1,_,_,m in zip(concentration, response_dead, response_malformation, response_normal, num_subject)]
data_L2 = [(x, m-y1, y2) for x,y1,y2,_,m in zip(concentration, response_dead, response_malformation, response_normal, num_subject)]

def logit_posterior_with_flat_prior(beta, data):
    log_post = 0
    for (x, m, y) in data:
        nu = beta[0] + beta[1]*x
        log_post += (y*nu - m*log(1+exp(nu)))
    return log_post

def logit_posterior_with_normal_prior(beta, data, sigma1, sigma2):
    log_post = logit_posterior_with_flat_prior(beta, data)
    log_post += (-0.5*beta[0]**2/(sigma1**2) - 0.5*beta[1]**2/(sigma2**2))
    return log_post

def symmetric_proposal_placeholder(from_smpl, to_smpl):
    #for log_proposal
    return 0

def normal_proposal_sampler(from_smpl, proposal_sigma_vec):
    return [normalvariate(x, proposal_sigma_vec[i]) for i,x in enumerate(from_smpl)]

def pi_plot(posterior_samples_L1, posterior_samples_L2, set_x_axis, show=True): #depend on data (not first-class function!)
    def inv_logit_link(x, params):
        # params: [beta1, beta2]
        nu = params[0]+params[1]*x
        return exp((nu))/(1+exp(nu))

    x_grid = np.arange(set_x_axis[0], set_x_axis[1], set_x_axis[2])
    pi1_lwr = []
    pi1_med = []
    pi1_upr = []
    pi1_avg = []
    pi2_lwr = []
    pi2_med = []
    pi2_upr = []
    pi2_avg = []
    pi3_lwr = []
    pi3_med = []
    pi3_upr = []
    pi3_avg = []
    for x in x_grid:
        rho1_samples_at_x = np.array([inv_logit_link(x, params) for params in posterior_samples_L1])
        rho2_samples_at_x = np.array([inv_logit_link(x, params) for params in posterior_samples_L2])
        
        pi1_samples_at_x = rho1_samples_at_x
        pi2_samples_at_x = rho2_samples_at_x * (1 - pi1_samples_at_x)
        pi3_samples_at_x = 1 - pi1_samples_at_x - pi2_samples_at_x

        pi1_lwr.append(float(np.quantile(pi1_samples_at_x, 0.025)))
        pi1_med.append(float(np.quantile(pi1_samples_at_x, 0.5)))
        pi1_upr.append(float(np.quantile(pi1_samples_at_x, 0.975)))
        pi1_avg.append(float(np.mean(pi1_samples_at_x)))
        pi2_lwr.append(float(np.quantile(pi2_samples_at_x, 0.025)))
        pi2_med.append(float(np.quantile(pi2_samples_at_x, 0.5)))
        pi2_upr.append(float(np.quantile(pi2_samples_at_x, 0.975)))
        pi2_avg.append(float(np.mean(pi2_samples_at_x)))
        pi3_lwr.append(float(np.quantile(pi3_samples_at_x, 0.025)))
        pi3_med.append(float(np.quantile(pi3_samples_at_x, 0.5)))
        pi3_upr.append(float(np.quantile(pi3_samples_at_x, 0.975)))
        pi3_avg.append(float(np.mean(pi3_samples_at_x)))

    plt.ylim([0, 1.05])
    plt.plot(x_grid, pi1_avg, color="black", linestyle="solid")
    plt.plot(x_grid, pi1_med, color="black", linestyle="dashed")
    plt.plot(x_grid, pi1_lwr, color="grey")
    plt.plot(x_grid, pi1_upr, color="grey")
    plt.plot(x_grid, pi2_avg, color="red", linestyle="solid")
    plt.plot(x_grid, pi2_med, color="red", linestyle="dashed")
    plt.plot(x_grid, pi2_lwr, color="grey")
    plt.plot(x_grid, pi2_upr, color="grey")
    plt.plot(x_grid, pi3_avg, color="blue", linestyle="solid")
    plt.plot(x_grid, pi3_med, color="blue", linestyle="dashed")
    plt.plot(x_grid, pi3_lwr, color="grey")
    plt.plot(x_grid, pi3_upr, color="grey")
    
    plt.plot([x for x in concentration], [y1/m for m, y1 in zip(num_subject, response_dead)], marker="o", color="black")
    plt.plot([x for x in concentration], [y2/m for m, y2 in zip(num_subject, response_malformation)], marker="o", color="red")
    plt.plot([x for x in concentration], [y3/m for m, y3 in zip(num_subject, response_normal)], marker="o", color="blue")
    if show:
        plt.show()

#L1_part_fit
part_L1_initial = [0,0]
part_L1_inst = MCMC_Core.MCMC_MH(
                partial(logit_posterior_with_normal_prior, data=data_L1, sigma1=10, sigma2=10),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.15766, 0.00043]), #from 4b
                part_L1_initial)
part_L1_inst.generate_samples(100000, print_iter_cycle=20000)
part_L1_diag = MCMC_Core.MCMC_Diag()
part_L1_diag.set_mc_sample_from_MCMC_instance(part_L1_inst)
part_L1_diag.set_variable_names(["alpha1", "beta1"])
part_L1_diag.burnin(20000)
part_L1_diag.thinning(5)
part_L1_diag.print_summaries(6)
part_L1_diag.show_traceplot((1,2))
part_L1_diag.show_acf(30, (1,2))
part_L1_diag.show_hist((1,2))

#L2 part fit
part_L2_initial = [0,0]
part_L2_inst = MCMC_Core.MCMC_MH(
                partial(logit_posterior_with_normal_prior, data=data_L2, sigma1=10, sigma2=10),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.33225, 0.00123]), #from 4b
                part_L2_initial)
part_L2_inst.generate_samples(100000, print_iter_cycle=20000)
part_L2_diag = MCMC_Core.MCMC_Diag()
part_L2_diag.set_mc_sample_from_MCMC_instance(part_L2_inst)
part_L2_diag.set_variable_names(["alpha2", "beta2"])
part_L2_diag.burnin(20000)
part_L2_diag.thinning(5)
part_L2_diag.print_summaries(6)
part_L2_diag.show_traceplot((1,2))
part_L2_diag.show_acf(30, (1,2))
part_L2_diag.show_hist((1,2))

#pi-plot
pi_plot(part_L1_diag.MC_sample, part_L2_diag.MC_sample, (0, 601, 2))