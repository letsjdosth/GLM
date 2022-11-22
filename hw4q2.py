from random import seed, normalvariate
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from pyBayes import MCMC_Core

seed(20221120)

#data
class AlligatorFoodChoice:
    def __init__(self) -> None:
        male_string = "1.30,I;1.80,F;1.32,F;1.85,F;1.32,F;1.93,I;1.40,F;1.93,F;1.42,I;1.98,I;1.42,F;2.03,F;1.47,I;2.03,F;1.47,F;2.31,F;1.50,I;2.36,F;1.52,I;2.46,F;1.63,I;3.25,O;1.65,O;3.28,O;1.65,O;3.33,F;1.65,I;3.56,F;1.65,F;3.58,F;1.68,F;3.66,F;1.70,I;3.68,O;1.73,O;3.71,F;1.78,F;3.89,F;1.78,O"
        female_string="1.24,I;2.56,O;1.30,I;2.67,F;1.45,I;2.72,I;1.45,O;2.79,F;1.55,I;2.84,F;1.60,I;1.60,I;1.65,F;1.78,I;1.78,O;1.80,I;1.88,I;2.16,F;2.26,F;2.31,F;2.36,F;2.39,F;2.41,F;2.44,F"
        male_split = [tuple(i.split(",")) for i in male_string.split(";")]
        female_split = [tuple(i.split(",")) for i in female_string.split(";")]

        self.y_length_gender = [(choice, float(length), 0) for length, choice in male_split] + [(choice, float(length), 1) for length, choice in female_split]
        self.y_length = [(choice, float(length)) for length, choice in male_split] + [(choice, float(length)) for length, choice in female_split]
        #gender: 0: male, 1: female


data_inst = AlligatorFoodChoice()
# print(data_inst.y_length)


def symmetric_proposal_placeholder(from_smpl, to_smpl):
    #for log_proposal
    return 0

def normal_proposal_sampler(from_smpl, proposal_sigma_vec):
    return [normalvariate(x, proposal_sigma_vec[i]) for i,x in enumerate(from_smpl)]

def log_posterior_with_flat_prior(beta, y_x):
    #beta: [beta_I_0, beta_I_1, (beta_I_2), beta_O_1, beta_O_2, (beta_O,2)]
    beta_dim = len(beta)//2
    beta_I = np.array(beta[0:beta_dim], dtype=float)
    beta_O = np.array(beta[beta_dim:len(beta)], dtype=float)
    
    log_post = 0
    for pt in y_x:
        y = pt[0]
        x = np.array([1]+list(pt[1:]), dtype=float)
        log_post -= np.log(1 + np.exp(np.dot(x, beta_I)) + np.exp(np.dot(x, beta_O)))
        if y == 'I':
            log_post += np.dot(x, beta_I)
        elif y == 'O':
            log_post += np.dot(x, beta_O)
        else:
            log_post += 0
    return log_post

def log_posterior_with_normal_prior(beta, y_x, sigma_vec):
    log_post = log_posterior_with_flat_prior(beta, y_x)
    log_post -= sum([b**2/(2*s**2) for b,s in zip(beta, sigma_vec)])
    return log_post

def pi_plot(posterior_samples, set_x_axis, gender=None, show=True): #depend on data (not first-class function!)
    beta_dim = len(posterior_samples[0])//2
    
    x_grid = np.arange(set_x_axis[0], set_x_axis[1], set_x_axis[2])
    pi1_lwr = [] #I
    pi1_med = []
    pi1_upr = []
    pi1_avg = []
    pi2_lwr = [] #O
    pi2_med = []
    pi2_upr = []
    pi2_avg = []
    pi3_lwr = [] #F
    pi3_med = []
    pi3_upr = []
    pi3_avg = []
    for x in x_grid:
        pi1_samples_at_x = []
        pi2_samples_at_x = []
        pi3_samples_at_x = []
        x_with_1 = None
        if gender is None:
            x_with_1 = np.array([1, x], dtype=float)
        elif gender==0:
            x_with_1 = np.array([1, x, 0], dtype=float)
        else: #gender==1
            x_with_1 = np.array([1, x, 1], dtype=float)

        for beta in posterior_samples:
            beta_I = np.array(beta[0:beta_dim], dtype=float)
            beta_O = np.array(beta[beta_dim:len(beta)], dtype=float)
            denom = 1+np.exp(np.dot(x_with_1, beta_I))+np.exp(np.dot(x_with_1, beta_O))

            pi1_samples_at_x.append(np.exp(np.dot(x_with_1, beta_I))/denom)
            pi2_samples_at_x.append(np.exp(np.dot(x_with_1, beta_O))/denom)
            pi3_samples_at_x.append(1/denom)

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
    plt.plot(x_grid, pi1_avg, color="black", linestyle="solid") #I
    plt.plot(x_grid, pi1_med, color="black", linestyle="dashed")
    plt.plot(x_grid, pi1_lwr, color="grey")
    plt.plot(x_grid, pi1_upr, color="grey")
    plt.plot(x_grid, pi2_avg, color="red", linestyle="solid") #O
    plt.plot(x_grid, pi2_med, color="red", linestyle="dashed")
    plt.plot(x_grid, pi2_lwr, color="grey")
    plt.plot(x_grid, pi2_upr, color="grey")
    plt.plot(x_grid, pi3_avg, color="blue", linestyle="solid") #F
    plt.plot(x_grid, pi3_med, color="blue", linestyle="dashed")
    plt.plot(x_grid, pi3_lwr, color="grey")
    plt.plot(x_grid, pi3_upr, color="grey")
    
    if show:
        plt.show()


# # 2a
length_initial = [0,0,0,0]
length_inst = MCMC_Core.MCMC_MH(
                partial(log_posterior_with_normal_prior, y_x=data_inst.y_length, sigma_vec=(10,10,10,10)),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.6, 0.6, 0.4, 0.25]),
                length_initial)
length_inst.generate_samples(50000, print_iter_cycle=5000)
length_diag = MCMC_Core.MCMC_Diag()
length_diag.set_mc_sample_from_MCMC_instance(length_inst)
length_diag.set_variable_names(["beta_I_0", "beta_I_1", "beta_O_0", "beta_O_1"])
length_diag.burnin(5000)
length_diag.thinning(5)
length_diag.print_summaries(6)
length_diag.show_traceplot((2,2))
length_diag.show_acf(30, (2,2))
length_diag.show_hist((2,2))

pi_plot(length_diag.MC_sample, (1.2, 3.0, 0.1), None)


# 2b
full_initial = [0,0,0,0,0,0]
full_inst = MCMC_Core.MCMC_MH(
                partial(log_posterior_with_normal_prior, y_x=data_inst.y_length_gender, sigma_vec=(10,10,10,10,10,10)),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.9, 0.3, 0.3, 0.35, 0.2, 0.3]),
                full_initial)
full_inst.generate_samples(50000, print_iter_cycle=5000)
full_diag = MCMC_Core.MCMC_Diag()
full_diag.set_mc_sample_from_MCMC_instance(full_inst)
full_diag.set_variable_names(["beta_I_0", "beta_I_1", "beta_I_2", "beta_O_0", "beta_O_1", "beta_O_2"])
full_diag.burnin(5000)
full_diag.thinning(5)
full_diag.print_summaries(6)
full_diag.show_traceplot((2,3))
full_diag.show_acf(30, (2,3))
full_diag.show_hist((2,3))


pi_plot(full_diag.MC_sample, (1.2, 3.0, 0.1), 0) #gender: 0: male, 1: female
pi_plot(full_diag.MC_sample, (1.2, 3.0, 0.1), 1)
