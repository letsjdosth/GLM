from random import seed, normalvariate, gammavariate, uniform
from math import log, exp, lgamma, inf
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from pyBayes import MCMC_Core

seed(20221201)

x_fabric_length = [551, 651, 832, 375, 715, 868, 271, 630, 491, 372, 645, 441, 895, 458, 642, 492, 543, 842, 905, 542, 522, 122, 657, 170, 738, 371, 735, 749, 495, 716, 952, 417]
y_fabric_faults = [6, 4, 17, 9, 14, 8, 5, 7, 7, 7, 6, 8, 28, 4, 10, 4, 8, 9, 23, 9, 6, 1, 9, 4, 9, 14, 17, 10, 7, 3, 9, 2]
y_x_fabric = [(y, x) for y, x in zip(y_fabric_faults, x_fabric_length)]

def symmetric_proposal_placeholder(from_smpl, to_smpl):
    #for log_proposal
    return 0

def normal_proposal_sampler(from_smpl, proposal_sigma_vec):
    return [normalvariate(x, proposal_sigma_vec[i]) for i,x in enumerate(from_smpl)]

def unif_proposal_log_pdf(from_smpl, to_smpl, window):
    applied_window = [max(0, from_smpl-window/2), from_smpl+window/2]
    if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
        return -inf
    else:
        applied_window_len = applied_window[1] - applied_window[0]
        return 1/applied_window_len

def unif_proposal_sampler(from_smpl, window):
    applied_window = [max(0, from_smpl-window/2), from_smpl+window/2]
    return uniform(applied_window[0], applied_window[1])


#problem 1a
def q1a_log_posterior_with_flat_prior(beta, y_x):
    #beta: [beta1, beta2]
    log_post = 0
    for y, x in y_x:
        nu = beta[0] + beta[1] * x
        log_post += (y*nu - exp(nu))
    return log_post

q1a_initial = [0,0]
q1a_inst = MCMC_Core.MCMC_MH(
                partial(q1a_log_posterior_with_flat_prior, y_x=y_x_fabric),
                symmetric_proposal_placeholder,
                partial(normal_proposal_sampler, proposal_sigma_vec=[0.15, 0.0003]),
                q1a_initial)
q1a_inst.generate_samples(50000, print_iter_cycle=10000)
q1a_diag = MCMC_Core.MCMC_Diag()
q1a_diag.set_mc_sample_from_MCMC_instance(q1a_inst)
q1a_diag.set_variable_names(["beta_1", "beta_2"])
q1a_diag.burnin(5000)
q1a_diag.thinning(5)
q1a_diag.print_summaries(6)
q1a_diag.show_traceplot((1,2))
q1a_diag.show_acf(30, (1,2))
q1a_diag.show_hist((1,2))


#problem 1b
class Q1b_Gibbs(MCMC_Core.MCMC_Gibbs):
    def __init__(self, initial, y_x):
        self.MC_sample = [initial]
        self.y_x = y_x

    def _full_conditional_beta(self, last_param):
        new_param = last_param
        # 0                1                    2
        # [[beta1, beta2], [mu1, mu2,...,mu_n], lambda]
        def gibbs_beta_log_posterior_with_flat_prior(beta, mu_vec, lamb, y_x):
            log_post = 0
            for (_, x), mu in zip(y_x, mu_vec):
                nu = beta[0] + beta[1] * x
                log_post += ((-lamb)*(nu + mu * exp(-nu)))
            return log_post
        
        initial = last_param[0]
        mc_mh_inst = MCMC_Core.MCMC_MH(
                        partial(gibbs_beta_log_posterior_with_flat_prior, mu_vec=last_param[1], lamb=last_param[2], y_x=self.y_x),
                        symmetric_proposal_placeholder,
                        partial(normal_proposal_sampler, proposal_sigma_vec=[1.5, 0.002]),
                        initial)
        mc_mh_inst.generate_samples(2, verbose=False)
        new_beta = mc_mh_inst.MC_sample[-1]
        new_param[0] = new_beta
        return new_param

    def _full_conditional_mu(self, last_param):
        new_param = last_param
        # 0                1                    2
        # [[beta1, beta2], [mu1, mu2,...,mu_n], lambda]
        new_mu = last_param[1]
        for i, (y, x) in enumerate(self.y_x):
            nu = last_param[0][0] + last_param[0][1] * x
            lamb = last_param[2]
            new_mu[i] = gammavariate(lamb+y, 1+lamb*exp(-nu))
        new_param[1] = new_mu
        return new_param

    def _full_conditional_lambda(self, last_param):
        new_param = last_param
        # 0                1                    2
        # [[beta1, beta2], [mu1, mu2,...,mu_n], lambda]
        def gibbs_lambda_log_posterior_with_1_prior(lamb, beta, mu_vec, y_x):
            log_post = 0
            for (_, x), mu in zip(y_x, mu_vec):
                nu = beta[0] + beta[1] * x
                log_post += (log(mu) - nu - mu*exp(-nu))
            log_post += (len(y_x)*log(lamb))
            log_post *= lamb
            log_post -= (len(y_x)*lgamma(lamb))
            return log_post

        def gibbs_lambda_log_posterior_with_invquad_prior(lamb, beta, mu_vec, y_x):
            log_post = gibbs_lambda_log_posterior_with_1_prior(lamb, beta, mu_vec, y_x)
            log_post += (-2*log(lamb+1))
            return log_post
        
        initial = last_param[2]
        window = 10
        mc_mh_inst = MCMC_Core.MCMC_MH(
                        partial(gibbs_lambda_log_posterior_with_invquad_prior, beta = last_param[0], mu_vec=last_param[1], y_x=self.y_x),
                        partial(unif_proposal_log_pdf, window=window),
                        partial(unif_proposal_sampler, window=window),
                        initial)
        mc_mh_inst.generate_samples(2, verbose=False)
        new_lamb = mc_mh_inst.MC_sample[-1]
        new_param[2] = new_lamb
        return new_param

    def sampler(self, **kwargs):
        # [[beta1, beta2], [mu1, mu2,...,mu_n], lambda]
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self._full_conditional_beta(new)
        new = self._full_conditional_mu(new)
        new = self._full_conditional_lambda(new)
        self.MC_sample.append(new)


q1b_initial = [[0,0], [0 for _ in range(len(y_x_fabric))], 0.5]
q1b_inst = Q1b_Gibbs(q1b_initial, y_x_fabric)
q1b_inst.generate_samples(100000, print_iter_cycle=20000)

q1b_diag1 = MCMC_Core.MCMC_Diag()
q1b_MC_samples1 = [t[0] + [t[2]] for t in q1b_inst.MC_sample]
q1b_diag1.set_mc_samples_from_list(q1b_MC_samples1)
q1b_diag1.set_variable_names(["beta_1", "beta_2", "lambda"])
q1b_diag1.burnin(5000)
q1b_diag1.thinning(2)
q1b_diag1.print_summaries(6)
q1b_diag1.show_traceplot((1,3))
q1b_diag1.show_acf(30, (1,3))
q1b_diag1.show_hist((1,3))

q1b_diag2 = MCMC_Core.MCMC_Diag()
q1b_MC_samples1 = [t[1] for t in q1b_inst.MC_sample]
q1b_diag2.set_mc_samples_from_list(q1b_MC_samples1)
q1b_diag2.set_variable_names(["mu"+str(i) for i in range(len(y_x_fabric))])
q1b_diag2.burnin(5000)
q1b_diag2.thinning(2)
q1b_diag2.print_summaries(6)
q1b_diag2.show_traceplot((1,3), [0,1,2])
q1b_diag2.show_acf(30, (1,3), [0,1,2])
q1b_diag2.show_hist((1,3), [0,1,2])
