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

# q1a_initial = [0,0]
# q1a_inst = MCMC_Core.MCMC_MH(
#                 partial(q1a_log_posterior_with_flat_prior, y_x=y_x_fabric),
#                 symmetric_proposal_placeholder,
#                 partial(normal_proposal_sampler, proposal_sigma_vec=[0.15, 0.0003]),
#                 q1a_initial)
# q1a_inst.generate_samples(50000, print_iter_cycle=10000)
# q1a_diag = MCMC_Core.MCMC_Diag()
# q1a_diag.set_mc_sample_from_MCMC_instance(q1a_inst)
# q1a_diag.set_variable_names(["beta_1", "beta_2"])
# q1a_diag.burnin(5000)
# q1a_diag.thinning(5)
# q1a_diag.print_summaries(6)
# q1a_diag.show_traceplot((1,2))
# q1a_diag.show_acf(30, (1,2))
# q1a_diag.show_hist((1,2))


def q1a_mu_plot(posterior_samples, set_x_axis, show=True): #depend on data (not first-class function!)
    x_grid = np.arange(set_x_axis[0], set_x_axis[1], set_x_axis[2])
    mu_lwr = [] #I
    mu_med = []
    mu_upr = []
    mu_avg = []

    for x in x_grid:
        mu_samples_at_x = []

        for beta in posterior_samples:
            mu_samples_at_x.append(exp(beta[0]+beta[1]*x))

        mu_lwr.append(float(np.quantile(mu_samples_at_x, 0.025)))
        mu_med.append(float(np.quantile(mu_samples_at_x, 0.5)))
        mu_upr.append(float(np.quantile(mu_samples_at_x, 0.975)))
        mu_avg.append(float(np.mean(mu_samples_at_x)))

    plt.ylim([0, 30])
    plt.plot(x_grid, mu_avg, color="black", linestyle="solid")
    plt.plot(x_grid, mu_med, color="black", linestyle="dashed")
    plt.plot(x_grid, mu_lwr, color="grey")
    plt.plot(x_grid, mu_upr, color="grey")

    plt.plot(x_fabric_length, y_fabric_faults, marker="o", color="black", linestyle="none")
    
    if show:
        plt.show()

def q1a_generate_predictive_samples(posterior_samples): #depend on data (not first-class function!)
    predictive_samples = []
    for x in x_fabric_length:
        pred_samples_at_x = []
        for beta in posterior_samples:
            mu_at_x = exp(beta[0]+beta[1]*x)
            pred_samples_at_x.append(np.random.poisson(mu_at_x))
        predictive_samples.append(pred_samples_at_x)
    return predictive_samples

def post_pred_resid_plot(predictive_samples): #depend on data (not first-class function!)
    x_grid = x_fabric_length
    boxplot_frame = []
    for i, (obs_y, _) in enumerate(y_x_fabric):
        res_samples_at_x = [y_new - obs_y for y_new in predictive_samples[i]]
        boxplot_frame.append(res_samples_at_x)

    plt.figure(figsize=(12,5))
    plt.boxplot(np.transpose(boxplot_frame), labels=x_grid)
    plt.axhline(0)
    plt.tight_layout()
    plt.show()


def loss_L_measure(predictive_samples, k): #depend on data (not first-class function!)
    loss = 0
    for i, (obs_y, _) in enumerate(y_x_fabric):
        predictive_var_at_x = np.var(predictive_samples[i])
        predictive_mean_at_x = np.mean(predictive_samples[i])
        loss += (predictive_var_at_x + (k/(k+1))*(obs_y-predictive_mean_at_x)**2)
    return loss


# q1a_mu_plot(q1a_diag.MC_sample, (100, 1000, 1))

# q1a_predictive_samples = q1a_generate_predictive_samples(q1a_diag.MC_sample)
# post_pred_resid_plot(q1a_predictive_samples)
# print("quadratic loss measure, k=1 :", loss_L_measure(q1a_predictive_samples, 1))
# print("quadratic loss measure, k=2 :", loss_L_measure(q1a_predictive_samples, 2))
# print("quadratic loss measure, k=5 :", loss_L_measure(q1a_predictive_samples, 5))
# print("quadratic loss measure, k=10 :", loss_L_measure(q1a_predictive_samples, 10))


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
                        partial(normal_proposal_sampler, proposal_sigma_vec=[0.6, 0.0005]),
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
            new_mu[i] = gammavariate(lamb+y, 1/(1+lamb*exp(-nu)))
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
        
        def gibbs_lambda_log_posterior_with_gamma_prior(lamb, beta, mu_vec, y_x):
            log_post = gibbs_lambda_log_posterior_with_1_prior(lamb, beta, mu_vec, y_x)
            a = 2
            b = 2
            log_post += ((a-1)*log(lamb) - b*lamb)
            return log_post

        initial = last_param[2]
        window = 10
        mc_mh_inst = MCMC_Core.MCMC_MH(
                        partial(gibbs_lambda_log_posterior_with_invquad_prior, beta = last_param[0], mu_vec=last_param[1], y_x=self.y_x), #prior sensitivity analysis!
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
q1b_inst.generate_samples(50000, print_iter_cycle=10000)

q1b_diag1 = MCMC_Core.MCMC_Diag()
q1b_MC_samples1 = [t[0] + [t[2]] for t in q1b_inst.MC_sample]
q1b_diag1.set_mc_samples_from_list(q1b_MC_samples1)
q1b_diag1.set_variable_names(["beta_1", "beta_2", "lambda"])
q1b_diag1.burnin(5000)
q1b_diag1.thinning(5)
q1b_diag1.print_summaries(6)
q1b_diag1.show_traceplot((1,3))
q1b_diag1.show_acf(30, (1,3))
q1b_diag1.show_hist((1,3))

q1b_diag2 = MCMC_Core.MCMC_Diag()
q1b_MC_samples1 = [t[1] for t in q1b_inst.MC_sample]
q1b_diag2.set_mc_samples_from_list(q1b_MC_samples1)
q1b_diag2.set_variable_names(["mu"+str(i) for i in range(len(y_x_fabric))])
q1b_diag2.burnin(5000)
q1b_diag2.thinning(5)
q1b_diag2.print_summaries(6)
q1b_diag2.show_traceplot((1,3), [0,1,2])
q1b_diag2.show_acf(30, (1,3), [0,1,2])
q1b_diag2.show_hist((1,3), [0,1,2])



def q1b_mu_plot(posterior_beta_lambda_samples, set_x_axis, show=True): #depend on data (not first-class function!)
    x_grid = np.arange(set_x_axis[0], set_x_axis[1], set_x_axis[2])
    mu_lwr = []
    mu_med = []
    mu_upr = []
    mu_avg = []

    for x in x_grid:
        mu_samples_at_x = []

        for beta_lambda in posterior_beta_lambda_samples:
            lamb = beta_lambda[2]
            gamma_at_x = exp(beta_lambda[0]+beta_lambda[1]*x)
            mu_samples_at_x.append(gammavariate(lamb, 1/(lamb/gamma_at_x)))

        mu_lwr.append(float(np.quantile(mu_samples_at_x, 0.025)))
        mu_med.append(float(np.quantile(mu_samples_at_x, 0.5)))
        mu_upr.append(float(np.quantile(mu_samples_at_x, 0.975)))
        mu_avg.append(float(np.mean(mu_samples_at_x)))

    plt.ylim([0, 30])
    plt.plot(x_grid, mu_avg, color="black", linestyle="solid")
    plt.plot(x_grid, mu_med, color="black", linestyle="dashed")
    plt.plot(x_grid, mu_lwr, color="grey")
    plt.plot(x_grid, mu_upr, color="grey")

    plt.plot(x_fabric_length, y_fabric_faults, marker="o", color="black", linestyle="none")
    
    if show:
        plt.show()

def q1b_generate_predictive_samples(posterior_mu_samples): #depend on data (not first-class function!)
    predictive_samples = [[] for _ in range(len(x_fabric_length))]
    for mu in posterior_mu_samples:
        for i, m_i in enumerate(mu):
            predictive_samples[i].append(np.random.poisson(m_i))
    return predictive_samples


q1b_mu_plot(q1b_diag1.MC_sample, (100, 1000, 1))

q1b_predictive_samples = q1b_generate_predictive_samples(q1b_diag2.MC_sample)
post_pred_resid_plot(q1b_predictive_samples)
print("quadratic loss measure, k=1 :", loss_L_measure(q1b_predictive_samples, 1))
print("quadratic loss measure, k=2 :", loss_L_measure(q1b_predictive_samples, 2))
print("quadratic loss measure, k=5 :", loss_L_measure(q1b_predictive_samples, 5))
print("quadratic loss measure, k=10 :", loss_L_measure(q1b_predictive_samples, 10))
