import numpy as np
import matplotlib.pyplot as plt

def simplified_NRe_optimizer(initial, score, expected_infomation, data, tolerance = 0.001):
    optimized_seq = [initial]
    
    while(True):
        last = optimized_seq[-1]
        new = last + np.linalg.inv(expected_infomation(last, data)) @ score(last, data)
        optimized_seq.append(new)
        if abs(last-new)<tolerance:
            break
    return optimized_seq[-1]

def simplified_NRo_optimizer(initial, score, obs_infomation, data, tolerance = 0.001):
    optimized_seq = [initial]
    
    while(True):
        last = optimized_seq[-1]
        new = last - np.linalg.inv(obs_infomation(last, data)) @ score(last, data)
        optimized_seq.append(new)
        if abs(last-new)<tolerance:
            break
    return optimized_seq[-1]

# cauchy(0,1) case
def cauchy_theta_1_log_likelihood(theta, data):
    theta = theta[0]
    lik = -len(data)*np.log(np.pi)
    for y in data:
        lik -= np.log(1+(y-theta)**2)
    return lik

def cauchy_theta_1_score(theta, data):
    theta = theta[0]
    score = 0
    for y in data:
        score += (2*(y-theta) / (1+(y-theta)**2))
    return np.array([score])

def cauchy_theta_1_hessian(theta, data):
    theta = theta[0]
    hessian = 0
    for y in data:
        hessian -= 2*(1-y-theta)**2 / (1+(y-theta)**2)**2
    return np.array([[hessian]])

def cauchy_theta_1_expected_information(_, data):
    return np.array([[len(data)/2]])

# problem 3b
y_3b = [-0.774, 0.597, 7.575, 0.397, -0.865, -0.318, -0.125, 0.961, 1.039]
initial_3b = np.array([0])
mle_NRe_3b = simplified_NRe_optimizer(initial_3b, cauchy_theta_1_score, cauchy_theta_1_expected_information, y_3b)
mle_NRo_3b = simplified_NRo_optimizer(initial_3b, cauchy_theta_1_score, cauchy_theta_1_hessian, y_3b)
print(mle_NRe_3b, mle_NRo_3b)

#plot
grid_3b = np.arange(-20, 20, 0.1)
log_likelihood_on_grid_3b = [cauchy_theta_1_log_likelihood([x], y_3b) for x in grid_3b]
plt.plot(grid_3b, log_likelihood_on_grid_3b)
plt.axvline(mle_NRe_3b, color="red", linestyle="dashed", linewidth=0.8)
plt.axvline(mle_NRo_3b, color="blue", linestyle="dashed", linewidth=0.8)
plt.show()


#problem 3c
y_3c = [0, 5, 9]
initial_3c_vec = [np.array([-1]), np.array([4.67]), np.array([10])]

grid_3c = np.arange(-20, 20, 0.1)
log_likelihood_on_grid_3c = [cauchy_theta_1_log_likelihood([x], y_3c) for x in grid_3c]
plt.plot(grid_3c, log_likelihood_on_grid_3c)

for initial_3c in initial_3c_vec: 
    mle_NRe_3c = simplified_NRe_optimizer(initial_3c, cauchy_theta_1_score, cauchy_theta_1_expected_information, y_3c)
    mle_NRo_3c = simplified_NRo_optimizer(initial_3c, cauchy_theta_1_score, cauchy_theta_1_hessian, y_3c)
    print(mle_NRe_3c, mle_NRo_3c)

    plt.axvline(mle_NRe_3c, color="red", linestyle="dashed", linewidth=0.8)
    plt.axvline(mle_NRo_3c, color="blue", linestyle="dashed", linewidth=0.8)

plt.show()

