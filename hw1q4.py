import numpy as np
import matplotlib.pyplot as plt

def l2_norm(x,y):
    return (sum([(a-b)**2 for a,b in zip(x,y)]))**0.5


def simplified_NRe_optimizer(initial, score, expected_infomation, data, tolerance = 0.001):
    optimized_seq = [initial]
    
    while(True):
        last = optimized_seq[-1]
        new = last + np.linalg.inv(expected_infomation(last, data)) @ score(last, data)
        # print(new)
        optimized_seq.append(new)
        if l2_norm(last, new)<tolerance:
            break
    return optimized_seq[-1]

def simplified_NRo_optimizer(initial, score, obs_infomation, data, tolerance = 0.001):
    optimized_seq = [initial]
    
    while(True):
        last = optimized_seq[-1]
        new = last - np.linalg.inv(obs_infomation(last, data)) @ score(last, data)
        # print(new)
        optimized_seq.append(new)
        if l2_norm(last, new)<tolerance:
            break
    return optimized_seq[-1]


# problem 4: poisson glm
def pois_glm_log_likelihood(beta, data):
    beta1 = beta[0]
    beta2 = beta[1]

    lik = 0
    for (x, y) in data:
        lik += (y*(beta1+beta2*x)-np.exp(beta1+beta2*x)-sum([np.log(a) for a in range(1, y+1)]))
    return lik

def pois_glm_score(beta, data):
    beta1 = beta[0]
    beta2 = beta[1]

    score = np.array([0, 0], dtype='float64')
    for (x, y) in data:
        exp_term = np.exp(beta1+beta2*x)
        score += np.array([y-exp_term, x*y-x*exp_term])
    return score

def pois_glm_hessian(beta, data):
    beta1 = beta[0]
    beta2 = beta[1]
    hessian = np.array([[0,0],[0,0]], dtype='float64')
    for (x, y) in data:
        exp_term = np.exp(beta1+beta2*x)
        hessian -= (np.array([[1, x],
                            [x, x**2]])*exp_term)
    return hessian

def pois_glm_expected_information(beta, data):
    beta1 = beta[0]
    beta2 = beta[1]

    expected_info = np.array([[0,0],[0,0]], dtype='float64')
    for (x, _) in data:
        exp_term = np.exp(beta1+beta2*x)
        expected_info += (np.array([[1, x],
                                [x, (x**2)]])*exp_term)
    return expected_info

y_4 = [1, 6, 16, 23, 27, 39, 31, 30, 43, 51, 63, 70, 88, 97, 91, 104, 110, 113, 149, 159]
data_4 = [(np.log(i+1),y) for i, y in enumerate(y_4)]
initial_4 = np.array([10,1])# do not start [0, 0]
mle_NRe_4 = simplified_NRe_optimizer(initial_4, pois_glm_score, pois_glm_expected_information, data_4)
mle_NRo_4 = simplified_NRo_optimizer(initial_4, pois_glm_score, pois_glm_hessian, data_4)
print(mle_NRe_4, mle_NRo_4)


grid_1 = np.linspace(0.9, 1.1, 100)
grid_2 = np.linspace(1, 1.5, 100)
meshgrid_1, meshgrid_2 = np.meshgrid(grid_1, grid_2)
value_mat = np.zeros(meshgrid_1.shape)
for i in range(len(grid_1)):
    for j in range(len(grid_2)):
        value_mat[i,j] = pois_glm_log_likelihood([meshgrid_1[i,j], meshgrid_2[i,j]], data_4)
plt.contour(meshgrid_1, meshgrid_2, value_mat, levels=30)
plt.plot(mle_NRe_4[0],mle_NRe_4[1], 'ro')
plt.plot(mle_NRo_4[0],mle_NRo_4[1], 'bo')
plt.show()