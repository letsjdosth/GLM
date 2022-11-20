concentration <- c(0, 62.5, 125, 250, 500)
num_subject <- c(297, 242, 312, 299, 285)
response <- matrix(
    c(15, 1, 281,
    17, 0, 225,
    22, 7, 283,
    38, 59, 202,
    144, 132, 9), 5, 3, byrow=TRUE)


# L1 part fit
L1_y = response[,1]
L1_m_y = response[,2]+response[,3]
L1_fit = glm(cbind(L1_y, L1_m_y)~concentration, family="binomial")
summary(L1_fit)

# L2 part fit
L2_y = response[,2]
L2_m_y = response[,3]
L2_fit = glm(cbind(L2_y, L2_m_y)~concentration, family="binomial")
summary(L2_fit)


grid = seq(0, 601, 1)
inv_logit <- function(x, coeff_vec){
    alpha = coeff_vec[1]
    beta = coeff_vec[2]
    exp_val = exp(alpha+beta*x)
    return(exp_val/(1+exp_val))
}


rho1_on_grid = inv_logit(grid, coef(L1_fit))
rho2_on_grid = inv_logit(grid, coef(L2_fit))
pi1_on_grid = rho1_on_grid
pi2_on_grid = rho2_on_grid * (1 - pi1_on_grid)
pi3_on_grid = 1 - pi1_on_grid - pi2_on_grid
plot(grid, pi1_on_grid, type="l", ylim=c(0,1), main="pi")
points(grid, pi2_on_grid, type="l", col="red")
points(grid, pi3_on_grid, type="l", col="blue")
