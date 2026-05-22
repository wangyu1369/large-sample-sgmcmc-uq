#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import random
from scipy.stats import bernoulli
from utils import (linear_normal, 
                    nonlinear_exp, 
                    nonlinear_quadratic,
                    nonlinear_dependent_noise)


##############################################
#### Determine the dimension and sample size
##############################################     
        

dimension = 5
all_sample_size = 2000

######## dataset for mis-specified case

model = nonlinear_dependent_noise(N=all_sample_size, rho=0, p = dimension, sigma = 0.1, penalty_para = 0, 
                      theta_loc = 0, theta_sigma = 1, bias = 1)
    
sample_size = 1500
test_sample_size = all_sample_size-sample_size

x_raw_all, y_raw_all = model.generate_data()

y_raw_all = y_raw_all.reshape(y_raw_all.shape[1], )

x_raw_test = x_raw_all[:test_sample_size, :]
y_raw_test = y_raw_all[:test_sample_size]

x_raw = x_raw_all[test_sample_size:, :]
y_raw = y_raw_all[test_sample_size:]


#############################################
### find the smallest and largest components
#############################################
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Standardize the data (important for PCA)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(x_raw)

# # Perform PCA
# pca = PCA()
# pca.fit(X_scaled)

# # Get the explained variance (eigenvalues) and principal components (eigenvectors)
# explained_variance = pca.explained_variance_    # Eigenvalues
# principal_components = pca.components_          # Eigenvectors (Principal components)

# # Find the smallest and largest principal component
# smallest_component = principal_components[np.argmin(explained_variance)]
# largest_component = principal_components[np.argmax(explained_variance)]

# print("Smallest Principal Component (Eigenvector):")
# print(smallest_component)
# print("\nLargest Principal Component (Eigenvector):")
# print(largest_component)


###########################################
#### Compute MLE and required ingredients
###########################################

sample_size = x_raw.shape[0]
dimension = x_raw.shape[1]

theta_hat = np.linalg.inv(x_raw.transpose()@x_raw)@x_raw.transpose()@y_raw
mle = theta_hat
precondition_matirx_Gamma =  sample_size * np.linalg.inv(x_raw.transpose()@x_raw)
scaled_precondition_matirx_Gamma =  np.linalg.inv(x_raw.transpose()@x_raw)


J_inverse = precondition_matirx_Gamma
J = np.linalg.inv(J_inverse)

V = 0 

for i in range(sample_size):
    V += (y_raw[i]-x_raw[i].dot(mle))**2*(x_raw[i].reshape(dimension, 
                                                    1)@x_raw[i].reshape(dimension, 1).T)

V /= sample_size

sanwich_matrix = J_inverse @ V @ J_inverse
scaled_sanwich_matrix = sanwich_matrix/sample_size

cov_bagged_posterior = 0.5*precondition_matirx_Gamma + 0.5*sanwich_matrix
scaled_cov_bagged_posterior = cov_bagged_posterior/sample_size

predicted_y = x_raw @ mle
errors = y_raw-predicted_y
E_vector = (errors.T @ x_raw)/sample_size
E = E_vector.reshape(dimension, 1) @ E_vector.reshape(1, dimension)


A = (x_raw.transpose()@x_raw)/sample_size
variance_noise = np.var(errors)


def calculate_c1(cov):
    res = 0
    for i in range(sample_size):
        res += (x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)@cov@(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
    
    res /= sample_size
    res -=  A@cov@A
    return res

import numpy as np
from scipy.optimize import fsolve

def stationary_cov_noise(batch_size, cov):
    
    C1 = calculate_c1(cov = cov)
    
    C2 = 0

    for i in range(sample_size):
        C2 += errors[i]**2*(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
        
    C2 /= sample_size

    C2 -= E
    
    C = C1 + C2
    
    C /= batch_size
    
    return C

def fluction_equation(Lambda, batch_size, cov_flatten, J = J):
    cov = cov_flatten.reshape(dimension, dimension)
    C = stationary_cov_noise(batch_size = batch_size, cov = cov)
    equation = Lambda@J@cov + cov@J@Lambda - Lambda@C@Lambda-Lambda@J@cov@J@Lambda
    return equation.flatten()

def stationary_cov_noise_l2(batch_size, cov, gamma):
    
    C1 = calculate_c1(cov = cov)
    
    C2 = 0

    for i in range(sample_size):
        C2 += errors[i]**2*(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
        
    C2 /= sample_size

    C2 -= (gamma**2/(sample_size**2)*np.outer(mle, mle))
    
    C = C1 + C2
    
    C /= batch_size
    
    return C

def fluction_equation_l2(Lambda, batch_size, cov_flatten, gamma, H):
    cov = cov_flatten.reshape(dimension, dimension)
    C = stationary_cov_noise_l2(batch_size = batch_size, cov = cov, gamma = gamma)
    equation = Lambda@H@cov + cov@H@Lambda - Lambda@C@Lambda-Lambda@H@cov@H@Lambda
    return equation.flatten()

def fluction_equation_discrete_l2(Lambda, batch_size, cov_flatten, gamma, H):
    cov = cov_flatten.reshape(dimension, dimension)
    C = H/batch_size
    equation = Lambda@H@cov + cov@H@Lambda - Lambda@C@Lambda-Lambda@H@cov@H@Lambda
    return equation.flatten()

# Initial guess (flattened version of a dxd matrix)
initial_guess = np.zeros((dimension, dimension)).flatten()

# Solve the system
solution_flat = fsolve(lambda x: fluction_equation(Lambda = 0.1*np.identity(dimension), batch_size = 1, cov_flatten = x), initial_guess)
    
theoretical_cov = solution_flat.reshape(dimension, dimension)

def fluction_equation_cts(Lambda, batch_size, cov_flatten, J=J):
    cov = cov_flatten.reshape(dimension, dimension)
    C = cov/batch_size
    equation = J@cov + cov@J - Lambda@C
    return equation.flatten()

def calculate_cov_discrete(Lambda, batch_size):
    G_mu = 2*np.identity(dimension) - (1+ 1/batch_size)*Lambda@A
    kappa_mu = np.trace(Lambda@A@np.linalg.inv(G_mu))/(1-(1/batch_size)*np.trace(Lambda@A@np.linalg.inv(G_mu)))
    return variance_noise/batch_size*(1+kappa_mu)*Lambda@np.linalg.inv(G_mu)

def calculate_cov_discrete_l2(Lambda, batch_size, gamma):
    K = A + gamma*np.identity(dimension)
    G = 2*np.identity(dimension)-Lambda*(K+np.linalg.inv(K)@A@A/batch_size)
    U = np.outer(mle, mle)
    trace_1 = np.trace(A@A@np.linalg.inv(K)@np.linalg.inv(G))
    kappa = trace_1/(1-Lambda/batch_size*trace_1)
    trace_2 = (gamma**2)*np.trace(A@A@A@np.linalg.inv(K@K@K)@np.linalg.inv(G)@U)
    trace_3 = np.trace(A@A@np.linalg.inv(K)@np.linalg.inv(G))
    r = trace_2/(1- Lambda/batch_size*trace_3)
    matrix = (gamma**2)*A@np.linalg.inv(K@K)@U
    return Lambda/batch_size*np.trace(matrix)*(1+Lambda*kappa/batch_size)*A@np.linalg.inv(K)@np.linalg.inv(G) + Lambda/batch_size*(matrix+Lambda*r/batch_size*A)@np.linalg.inv(K)@np.linalg.inv(G)

def test_loss_discrete_l2(Lambda, batch_size, gamma):
    K = A + gamma*np.identity(dimension)
    G = 2*np.identity(dimension)-Lambda*(K+np.linalg.inv(K)@A@A/batch_size)
    U = np.outer(mle, mle)
    trace_1 = np.trace(A@A@np.linalg.inv(K)@np.linalg.inv(G))
    kappa = trace_1/(1-Lambda/batch_size*trace_1)
    trace_2 = (gamma**2)*np.trace(A@A@A@np.linalg.inv(K@K@K)@np.linalg.inv(G)@U)
    trace_3 = np.trace(A@A@np.linalg.inv(K)@np.linalg.inv(G))
    r = trace_2/(1- Lambda/batch_size*trace_3)
    matrix = (gamma**2)*A@np.linalg.inv(K@K)@U
    return 0.5*Lambda/batch_size*(np.trace(matrix)*kappa+r) + 0.5*np.trace(matrix)
    
    
    

########################################
### SGD framework for linear regression
########################################


def sgd(n, batch_size, theta_0, lr_0, pre_matirx, 
        fixed_lr=True, 
         N = sample_size, precondition = False, w1=1, w2=1, 
         fixed_point=False, regulizer=False, penalty = 0):
    ret = np.zeros((n+1, dimension))
    ret[0,:] = theta_0
    if precondition:
        precondition_matirx = pre_matirx
    else:
        precondition_matirx = np.identity(dimension)
    for i in range(n):
        if fixed_lr:
            gamma = w1*lr_0
        else:
            gamma = w1*lr_0/(i+1)
        index = np.random.randint(0, N-1, batch_size)
        x_subsample =  x_raw[index, :]
        y_subsample = y_raw[index]
        
        delta_theta = 0
        pre_theta = ret[i,:]
        for j in range(batch_size):
            x_j = x_subsample[j,:].reshape(dimension, 1)
            if fixed_point:
                delta_theta += x_j @ x_j.T @ (pre_theta - theta_hat)
                
            else:
                delta_theta += x_j @ x_j.T @ pre_theta - x_subsample[j,:]*y_subsample[j]
        

        delta_theta *= (gamma/batch_size)
        delta_theta = precondition_matirx@delta_theta
        # noise = np.random.normal(0, np.sqrt(2*gamma/N), 6)
        
        if not regulizer:
            ret[i+1, :] = pre_theta-delta_theta
        else:
            ret[i+1, :] = (1-penalty*gamma/sample_size)*pre_theta-delta_theta
    return ret


def test_loss(x_test, y_test, mle, cov):
    
    M = x_test.shape[0]
    
    predicted_y_test = x_test @ mle
    
    errors = y_test-predicted_y_test
    loss_1  = np.mean(errors**2)
    loss_2 = 0
    
    for i in range(M):
        loss_2 += x_test[i].T@cov@x_test[i]
    loss_2 /= M                                                      
    
    return loss_1 + loss_2   

def calculate_H(J, Gamma):
    return J + Gamma/sample_size                                                 

####################################
### sample with fixed learning rate
####################################

b = 64
# b = int(sample_size*0.2)
lr_list = [0.3]
# lr_list = [0.05, 0.1, 0.15]
# lr_list = [50, 100, 200]

### set initilization

initial_state = mle
# initial_state = np.random.normal(0, 1, dimension)


num_epochs = 500


samples_record_sgd_path = {}
empirical_cov_path = {}
theoretical_cov_path = {}
countinuous_cov_path = {}
discrete_cov_path = {}


theoretical_test_loss_path = []
countinuous_test_loss_path = []
discrete_test_loss_path = []
empirical_test_loss_path = []


for l in range(len(lr_list)):
    # index = [x*n_iterations[l] for x in range(number_epochs+1)]
    num_iterations = num_epochs *int(sample_size/b)
    
    samples_path_sgd = sgd(n=num_iterations, batch_size = b, 
                           pre_matirx = np.identity(dimension),
                             theta_0=initial_state, lr_0 = lr_list[l], fixed_lr=True, w1=1, w2=1,
                             fixed_point=False,
                             precondition=True, regulizer=False)
    
    samples_record_sgd_path[l] = samples_path_sgd
    empirical_cov_path[l] = np.cov(samples_path_sgd.T)
    # cur_test_loss = 0
    # for i in range(samples_path_sgd.shape[0]):
    #     cur_test_loss += np.mean((y_raw_test - x_raw_test@samples_path_sgd[i].reshape(dimension, ))**2)
    
    # empirical_test_loss_path.append(cur_test_loss/samples_path_sgd.shape[0])
    
    # Initial guess (flattened version of a dxd matrix)
    initial_guess = np.identity(dimension).flatten()
    # initial_guess = np.zeros((dimension, dimension)).flatten()


    # Solve the system
    solution_flat = fsolve(lambda x: fluction_equation(Lambda = lr_list[l]*np.identity(dimension), 
                                                       batch_size = b, cov_flatten = x), initial_guess)
    
    ### compute the proposed theoretical covariance matrix
    theoretical_cov = solution_flat.reshape(dimension, dimension)
    cts_cov = lr_list[l]/b*np.identity(dimension)
    discrete_cov = calculate_cov_discrete(Lambda = lr_list[l]*np.identity(dimension), batch_size = b)
    
    ### compute the proposed theoretical test loss
    # theoretical_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = theoretical_cov))
    # countinuous_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = cts_cov))
    # discrete_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = discrete_cov))
    
    
    theoretical_cov_path[l] = theoretical_cov
    countinuous_cov_path[l] = cts_cov
    discrete_cov_path[l] = discrete_cov
    
# plt.plot(lr_list, empirical_test_loss_path,label= 'experiment')
# plt.plot(lr_list, theoretical_test_loss_path, label = 'this paper')
# plt.plot(lr_list, countinuous_test_loss_path, label = 'continuous-time')
# plt.plot(lr_list, discrete_test_loss_path, label = 'discrete-time')
# plt.legend()
    


# def log_density_normal(x, mu=mean_true, Sigma=cov_true):
#     """
#     log_density of target gaussian model
#     """
#     return multivariate_normal.logpdf(x, mean = mu, cov = Sigma)
            
# def log_density_normal_cts(x, mu=mean_true, Sigma=cov_cts):
#     """
#     log_density of target gaussian model
#     """
#     return multivariate_normal.logpdf(x, mean = mu, cov = Sigma)

# def log_density_normal_discrete(x, mu=mean_true, Sigma=cov_discrete):
#     """
#     log_density of target gaussian model
#     """
#     return multivariate_normal.logpdf(x, mean = mu, cov = Sigma)
            


# n_samples_for_plots = 1000

# plt.figure(figsize = (27, 8))
# plt.rcParams.update({'font.size': 24})
# xlim_setting = [0, 2]
# ylim_setting = [-1, 1.5]

# batch_size_index = 0


# xlim = xlim_setting
# ylim = ylim_setting

# xlist = np.linspace(*xlim, 1000)
# ylist = np.linspace(*ylim, 1000)
# X, Y = np.meshgrid(xlist, ylist)
# XY = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T


# zs = np.exp(log_density_normal_cts(XY))
# Z = zs.reshape(X.shape)
# cs_post = plt.contour(X, Y, Z, cmap='Reds_r', linestyles='solid')
# cs_post.collections[len(cs_post.collections)//2].set_label('Continuous-time')

# # for c in cs_post.collections:
# #     c.set_color('red')
    
# zs = np.exp(log_density_normal_discrete(XY))
# Z = zs.reshape(X.shape)

# cs_post = plt.contour(X, Y, Z, cmap='Blues_r', linestyles='solid')
# cs_post.collections[len(cs_post.collections)//2].set_label('large-sample+well-specified')

# # for c in cs_post.collections:
# #     c.set_color('blue')
    
# zs = np.exp(log_density_normal(XY))
# Z = zs.reshape(X.shape)
# cs_post = plt.contour(X, Y, Z, cmap='Greys_r', linestyles='solid')
# cs_post.collections[len(cs_post.collections)//2].set_label('Exact')

# # for c in cs_post.collections:
# #     c.set_color('black')
    
# # plt.plot(samples_record_further_improved_sgd_path[batch_size_index][-n_samples_for_plots:, index_1], 
# #           samples_record_further_improved_sgd_path[batch_size_index][-n_samples_for_plots:, index_2], '*', 
# #           alpha=0.8, color='red',label = 'SGD, batch size={}'.format(b[batch_size_index]))
# x_samples = samples_record_sgd_path[batch_size_index][-n_samples_for_plots:, index_1]
# y_samples = samples_record_sgd_path[batch_size_index][-n_samples_for_plots:, index_2]
# x_min, x_max = min(x_samples), max(x_samples)
# y_min, y_max = min(y_samples), max(y_samples)

# X_samples, Y_samples = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
# positions = np.vstack([X_samples.ravel(), Y_samples.ravel()])

# # values = np.vstack([x_samples, y_samples])
# # kernel = stats.gaussian_kde(values)
# # Z_approx = np.reshape(kernel(positions).T, X_samples.shape)


# # cs_approx = plt.contour(X_samples[-n_samples_for_plots:], Y_samples[-n_samples_for_plots:], Z_approx, cmap='jet')
# # cs_approx.collections[len(cs_approx.collections)//2].set_label('Preconditioned SGD')

# # for c in cs_approx.collections:
# #     c.set_color('red')
# #     c.set_alpha(0.5)

# plt.plot(x_samples[-n_samples_for_plots:], y_samples[-n_samples_for_plots:], '*', alpha=0.3, color='blue',
#          label = 'Iterates', markersize=6)
# plt.xlabel('$X_{1}$')
# plt.ylabel('$x_{2}$')
# plt.title(r'$\lambda$ ={}'.format(lr))
# plt.legend(loc='lower left')
# # plt.axis('off')

from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

#index_1 = 2
#index_2 = 9
index_1 = 0
index_2 = 1
mean_true = mle[[index_1, index_2]]


# Custom handler to draw an ellipse outline as a line in the legend
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create an ellipse object to be used in the legend
        center = width / 2 - xdescent, height / 2 - ydescent
        p = Ellipse(center, width=width, height=height, angle=0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def plot_ellipse(mean, cov, ax, method, linestyle, n_std=3.0, **kwargs):
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(cov)
    
    # Sorting by largest eigenvalue
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    # Lengths of the axes (scaled by n_std)
    width, height = 2 * n_std * np.sqrt(vals)
    
    # Orientation of the ellipse
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Create and add the ellipse to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      label = method, linewidth=3, linestyle = linestyle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def compute_limits(mean, cov, n_std=3.0):
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(cov)
    
    # Width and height of the ellipse (scaled by n_std)
    width, height = 2 * n_std * np.sqrt(vals)
    
    # Major axis direction
    angle = np.arctan2(*vecs[:, 0][::-1])
    
    # Compute the extents of the ellipse
    x_extents = np.array([mean[0] - width / 2, mean[0] + width / 2])
    y_extents = np.array([mean[1] - height / 2, mean[1] + height / 2])
    
    return x_extents, y_extents


# n_samples_for_plots = samples_record_sgd_path[l].shape[0]
n_samples_for_plots = 1000

###################################
### make single plot for single lr
###################################
# Plot settings
fig, ax = plt.subplots(figsize = (5,5))

index = 0
    
# empirical_true = [[empirical_cov_path[index][index_1, index_1], empirical_cov_path[index][index_1, index_2]],
#             [empirical_cov_path[index][index_2, index_1], empirical_cov_path[index][index_2, index_2]]]

cov_true = [[theoretical_cov_path[index][index_1, index_1], theoretical_cov_path[index][index_1, index_2]],
            [theoretical_cov_path[index][index_2, index_1], theoretical_cov_path[index][index_2, index_2]]]

cov_cts = [[countinuous_cov_path[index][index_1, index_1], countinuous_cov_path[index][index_1, index_2]],
            [countinuous_cov_path[index][index_2, index_1], countinuous_cov_path[index][index_2, index_2]]]

cov_discrete = [[discrete_cov_path[index][index_1, index_1], discrete_cov_path[index][index_1, index_2]],
            [discrete_cov_path[index][index_2, index_1], discrete_cov_path[index][index_2, index_2]]]

samples=np.random.multivariate_normal(mean = mle, cov=empirical_cov_path[index], size=n_samples_for_plots)

x_samples = samples[:, index_1]
y_samples = samples[:, index_2]



# Create a figure with 3 subplots in a row (1 row, 3 columns)

# ax.axhline(0, color='grey', lw=1)
# ax.axvline(0, color='grey', lw=1)

# Plotting the 3-sigma ellipse


plt.scatter(x_samples, y_samples, marker ='*', 
                    alpha=1, color='#bcbd22', s=100)

plot_ellipse(mean = mean_true, cov = cov_cts, ax = ax, n_std=3.0, 
             method = 'Continuous-time', linestyle = '-',
             edgecolor='black', facecolor='none')

plot_ellipse(mean = mean_true, cov = cov_discrete, ax = ax, n_std=3.0, 
             method = 'large-sample+well-specified', linestyle = '--',
             edgecolor='#1f77b4', facecolor='none')

plot_ellipse(mean = mean_true, cov = cov_true, ax = ax, method = 'Exact', 
             n_std=3.0, linestyle = '-.', 
             edgecolor='#d62728', facecolor='none')

# if index==0:
#     axes[index].set_xlim(-2, 2)
#     axes[index].set_ylim(-2, 2)
# elif index==1:
#     axes[index].set_xlim(-5, 5)
#     axes[index].set_ylim(-5, 5)
# else:
#     axes[index].set_xlim(-11, 11)
#     axes[index].set_ylim(-11, 11)
# # Custom lines to appear in the legend
# Compute the axis limits for each individual ellipse
x_extents_cts, y_extents_cts = compute_limits(mean_true, cov_cts, n_std=3.0)
x_extents_discrete, y_extents_discrete = compute_limits(mean_true, cov_discrete, n_std=3.0)
x_extents_true, y_extents_true = compute_limits(mean_true, cov_true, n_std=3.0)

# Combine the extents of all ellipses for this subplot to set axis limits
x_min = min(x_extents_cts[0], x_extents_discrete[0], x_extents_true[0])-0.2
x_max = max(x_extents_cts[1], x_extents_discrete[1], x_extents_true[1])+0.2
y_min = min(y_extents_cts[0], y_extents_discrete[0], y_extents_true[0])-0.2
y_max = max(y_extents_cts[1], y_extents_discrete[1], y_extents_true[1])+0.2

# Set the individual axis limits for this subplot
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
    


custom_lines = [
                Line2D([0], [0], color='black', lw=2, linestyle = '-'),
                Line2D([0], [0], color='#1f77b4', lw=2, linestyle = 'dotted'),
                Line2D([0], [0], color='#d62728', lw=2, linestyle = '-.'),
                Line2D([0], [0], marker='*', color='#bcbd22', lw=0, markersize=6)  # Point marker
                ]
# if index==0:
#     # Add a legend with straight lines instead of ellipses
#     ax.legend(custom_lines, ['Continuous-time', 'Large-sample+well-specified', 'Exact (this paper)', 'Iterates'], 
#                        fontsize=13)
    # Remove x and y axis ticks (numbers)
    # axes[index].set_xticks([])
    # axes[index].set_yticks([])
# ax.set_title(r'$\lambda$ ={}'.format(lr_list[index]))
# plt.show()
plt.tight_layout(h_pad=1)




#########################################
### make multiple plots for multiple lrs
########################################

# Create a figure with 3 subplots in a row (1 row, 3 columns)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for index in range(len(lr_list)): 
    
    # empirical_true = [[empirical_cov_path[index][index_1, index_1], empirical_cov_path[index][index_1, index_2]],
    #             [empirical_cov_path[index][index_2, index_1], empirical_cov_path[index][index_2, index_2]]]

    cov_true = [[theoretical_cov_path[index][index_1, index_1], theoretical_cov_path[index][index_1, index_2]],
                [theoretical_cov_path[index][index_2, index_1], theoretical_cov_path[index][index_2, index_2]]]
    
    cov_cts = [[countinuous_cov_path[index][index_1, index_1], countinuous_cov_path[index][index_1, index_2]],
                [countinuous_cov_path[index][index_2, index_1], countinuous_cov_path[index][index_2, index_2]]]
    
    cov_discrete = [[discrete_cov_path[index][index_1, index_1], discrete_cov_path[index][index_1, index_2]],
                [discrete_cov_path[index][index_2, index_1], discrete_cov_path[index][index_2, index_2]]]
    
    samples=np.random.multivariate_normal(mean = mle, cov=empirical_cov_path[index], size=n_samples_for_plots)
    
    x_samples = samples[:, index_1]
    y_samples = samples[:, index_2]
    
    

    # Create a figure with 3 subplots in a row (1 row, 3 columns)

    # ax.axhline(0, color='grey', lw=1)
    # ax.axvline(0, color='grey', lw=1)
    
    # Plotting the 3-sigma ellipse
    
    axes[index].scatter(x_samples, y_samples, marker ='*', 
                        alpha=1, color='#bcbd22', s=130)
    
    plot_ellipse(mean = mean_true, cov = cov_cts, ax = axes[index], n_std=3.0, 
                 method = 'Continuous-time', linestyle = '-',
                 edgecolor='black', facecolor='none')
    
    plot_ellipse(mean = mean_true, cov = cov_discrete, ax = axes[index], n_std=3.0, 
                 method = 'large-sample+well-specified', linestyle = '--',
                 edgecolor='#1f77b4', facecolor='none')
    
    plot_ellipse(mean = mean_true, cov = cov_true, ax = axes[index], method = 'Exact', 
                 n_std=3.0, linestyle = '-.', 
                 edgecolor='#d62728', facecolor='none')
    
    # if index==0:
    #     axes[index].set_xlim(-2, 2)
    #     axes[index].set_ylim(-2, 2)
    # elif index==1:
    #     axes[index].set_xlim(-5, 5)
    #     axes[index].set_ylim(-5, 5)
    # else:
    #     axes[index].set_xlim(-11, 11)
    #     axes[index].set_ylim(-11, 11)
    # # Custom lines to appear in the legend
    # Compute the axis limits for each individual ellipse
    x_extents_cts, y_extents_cts = compute_limits(mean_true, cov_cts, n_std=3.0)
    x_extents_discrete, y_extents_discrete = compute_limits(mean_true, cov_discrete, n_std=3.0)
    x_extents_true, y_extents_true = compute_limits(mean_true, cov_true, n_std=3.0)
    
    # Combine the extents of all ellipses for this subplot to set axis limits
    x_min = min(x_extents_cts[0], x_extents_discrete[0], x_extents_true[0])-0.1
    x_max = max(x_extents_cts[1], x_extents_discrete[1], x_extents_true[1])+0.1
    y_min = min(y_extents_cts[0], y_extents_discrete[0], y_extents_true[0])-0.1
    y_max = max(y_extents_cts[1], y_extents_discrete[1], y_extents_true[1])+0.1
    
    # Set the individual axis limits for this subplot
    axes[index].set_xlim(x_min, x_max)
    axes[index].set_ylim(y_min, y_max)
        
    
    
    custom_lines = [
                    Line2D([0], [0], color='black', lw=2, linestyle = '-'),
                    Line2D([0], [0], color='#1f77b4', lw=2, linestyle = 'dotted'),
                    Line2D([0], [0], color='#d62728', lw=2, linestyle = '-.'),
                    Line2D([0], [0], marker='*', color='#bcbd22', lw=0, markersize=6)  # Point marker
                    ]
    if index==0:
        # Add a legend with straight lines instead of ellipses
        axes[index].legend(custom_lines, ['Continuous-time', 'Large-sample+well-specified', 'Exact (this paper)', 'Iterates'], 
                           fontsize=13)
        # Remove x and y axis ticks (numbers)
        # axes[index].set_xticks([])
        # axes[index].set_yticks([])
    axes[index].set_title(r'$\lambda$ ={}'.format(lr_list[index]))
# plt.show()
plt.tight_layout(h_pad=1)



    
####################################
### sample with fixed learning rate
####################################

b = 16
# lr_list = [0.1, 0.5, 0.8]
lr = 1
penalty_list = [-300, -200, -150, -100, 0, 100, 200, 300, 400, 500, 800, 850, 900,950,1000]



### set initilization

initial_state = mle
# initial_state = np.random.normal(0, 1, dimension)


num_epochs = 200


samples_record_sgd_path = {}
theoretical_cov_path = {}
countinuous_cov_path = {}
discrete_cov_path = {}


theoretical_test_loss_path = []
countinuous_test_loss_path = []
discrete_test_loss_path = []
empirical_test_loss_path = []

empirical_test_loss_path_var = []


for l in range(len(penalty_list)):
    # index = [x*n_iterations[l] for x in range(number_epochs+1)]
    num_iterations = num_epochs *int(sample_size/b)

    
    samples_path_sgd = sgd(n=num_iterations, batch_size = b, 
                           pre_matirx = np.identity(dimension),
                             theta_0=initial_state, lr_0 = lr, fixed_lr=True, w1=1, w2=1,
                             fixed_point=False,
                             precondition=True, regulizer=True, penalty = penalty_list[l])
    
    ## compute H
    cur_H = J + penalty_list[l]*np.identity(dimension)/sample_size
    
    samples_record_sgd_path[l] = samples_path_sgd
    cur_test_loss = 0
    cur_test_loss_all = []
    for i in range(samples_path_sgd.shape[0]):
        cur_test_loss += np.mean((y_raw_test - x_raw_test@samples_path_sgd[i].reshape(dimension, ))**2)
        cur_test_loss_all.append(np.mean((y_raw_test - x_raw_test@samples_path_sgd[i].reshape(dimension, ))**2))
        
    empirical_test_loss_path.append(cur_test_loss/samples_path_sgd.shape[0])
    empirical_test_loss_path_var.append(np.std(cur_test_loss_all))
    
    # Initial guess (flattened version of a dxd matrix)
    initial_guess = np.identity(dimension).flatten()
    # initial_guess = np.zeros((dimension, dimension)).flatten()


    # Solve the system
    solution_flat = fsolve(lambda x: fluction_equation_l2(Lambda = lr*np.identity(dimension), 
                        batch_size = b, gamma = penalty_list[l], H=cur_H, cov_flatten = x), initial_guess)
    
    solution_flat_discrete = fsolve(lambda x: fluction_equation_discrete_l2(Lambda = lr*np.identity(dimension), 
                        batch_size = b, gamma = penalty_list[l], H=cur_H, cov_flatten = x), initial_guess)
    
    ### compute the proposed theoretical covariance matrix
    theoretical_cov = solution_flat.reshape(dimension, dimension)
    cts_cov = lr/b*np.identity(dimension)
    # discrete_cov = calculate_cov_discrete_l2(Lambda = lr, batch_size = b, gamma = penalty_list[l])
    discrete_cov = solution_flat_discrete.reshape(dimension, dimension)
    
    ### compute the proposed theoretical test loss
    theoretical_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = theoretical_cov))
    countinuous_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = cts_cov))
    discrete_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = discrete_cov))
    # discrete_test_loss_path.append(test_loss_discrete_l2(Lambda = lr, batch_size = b, gamma = penalty_list[l])/sample_size)
    
    theoretical_cov_path[l] = theoretical_cov
    countinuous_cov_path[l] = cts_cov
    discrete_cov_path[l] = discrete_cov


plt.figure(figsize=(7, 5))
plt.scatter(2*np.array(penalty_list)/sample_size, empirical_test_loss_path,label= 'experiment', color = 'red', marker = '*', s = 100)
plt.plot(2*np.array(penalty_list)/sample_size, theoretical_test_loss_path, label = 'Exact (this paper)', linestyle = '-')
plt.plot(2*np.array(penalty_list)/sample_size, countinuous_test_loss_path, label = 'continuous-time', linestyle = '-.')
plt.plot(2*np.array(penalty_list)/sample_size, discrete_test_loss_path, label = 'large-sample+well-specified', linestyle = '--')
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('$\gamma$', fontsize=13)
plt.ylabel('Test Loss', fontsize=13)
plt.legend(fontsize=15)
# Adjust size of the numbers on x and y axes (tick labels)
plt.xticks(fontsize=12)  # X-axis tick label size
plt.yticks(fontsize=12)  # Y-axis tick label size
# Plot the confidence interval as a shaded area
plt.fill_between(2*np.array(penalty_list)/sample_size, np.array(empirical_test_loss_path) - 2*np.array(empirical_test_loss_path_var), np.array(empirical_test_loss_path) + 2*np.array(empirical_test_loss_path_var), color='blue', alpha=0.2, label='95% CI')

plt.tight_layout(h_pad=1)




#####################################
### change with different batch size
#####################################

b_list = [1, 2, 4, 8, 16, 32, 64]
# lr_list = [0.1, 0.5, 0.8]
lr = 0.1
penalty = 0



### set initilization

initial_state = mle
# initial_state = np.random.normal(0, 1, dimension)


num_epochs = 200


samples_record_sgd_path = {}
theoretical_cov_path = {}
countinuous_cov_path = {}
discrete_cov_path = {}


theoretical_test_loss_path = []
countinuous_test_loss_path = []
discrete_test_loss_path = []
empirical_test_loss_path = []


for l in range(len(b_list)):
    # index = [x*n_iterations[l] for x in range(number_epochs+1)]
    num_iterations = num_epochs *int(sample_size/b_list[l])
    
    samples_path_sgd = sgd(n=num_iterations, batch_size = b_list[l], 
                           pre_matirx = np.identity(dimension),
                             theta_0=initial_state, lr_0 = lr, fixed_lr=True, w1=1, w2=1,
                             fixed_point=False,
                             precondition=True, regulizer=True, penalty = penalty)
    
    ## compute H
    cur_H = J + penalty*np.identity(dimension)/sample_size
    
    samples_record_sgd_path[l] = samples_path_sgd
    cur_test_loss = 0
    for i in range(samples_path_sgd.shape[0]):
        cur_test_loss += np.mean((y_raw_test - x_raw_test@samples_path_sgd[i].reshape(dimension, ))**2)
    
    empirical_test_loss_path.append(cur_test_loss/samples_path_sgd.shape[0])
    
    # Initial guess (flattened version of a dxd matrix)
    initial_guess = np.identity(dimension).flatten()
    # initial_guess = np.zeros((dimension, dimension)).flatten()


    # Solve the system
    solution_flat = fsolve(lambda x: fluction_equation_l2(Lambda = lr*np.identity(dimension), 
                        batch_size = b_list[l], gamma = penalty_list[l], H=cur_H, cov_flatten = x), initial_guess)
    
    solution_flat_discrete = fsolve(lambda x: fluction_equation_discrete_l2(Lambda = lr*np.identity(dimension), 
                        batch_size = b_list[l], gamma = penalty_list[l], H=cur_H, cov_flatten = x), initial_guess)
    
    ### compute the proposed theoretical covariance matrix
    theoretical_cov = solution_flat.reshape(dimension, dimension)
    cts_cov = lr/b_list[l]*np.identity(dimension)
    # discrete_cov = calculate_cov_discrete_l2(Lambda = lr, batch_size = b_list[l], gamma = penalty_list[l])
    discrete_cov = solution_flat_discrete.reshape(dimension, dimension)
    
    ### compute the proposed theoretical test loss
    theoretical_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = theoretical_cov))
    countinuous_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = cts_cov))
    discrete_test_loss_path.append(test_loss(x_test = x_raw_test, y_test = y_raw_test, mle = mle, cov = discrete_cov))
    # discrete_test_loss_path.append(test_loss_discrete_l2(Lambda = lr, batch_size = b_list[l], gamma = penalty_list[l])/sample_size)
    
    theoretical_cov_path[l] = theoretical_cov
    countinuous_cov_path[l] = cts_cov
    discrete_cov_path[l] = discrete_cov


plt.figure(figsize=(7, 5))
plt.scatter(b_list, empirical_test_loss_path,label= 'experiment', color = 'red', marker = '*', s = 100)
plt.plot(b_list, theoretical_test_loss_path, label = 'Exact (this paper)', linestyle = '-')
plt.plot(b_list, countinuous_test_loss_path, label = 'continuous-time', linestyle = '-.')
plt.plot(b_list, discrete_test_loss_path, label = 'large-sample+well-specified', linestyle = '--')
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Batch  size', fontsize=13)
plt.ylabel('Test Loss', fontsize=13)
# Adjust size of the numbers on x and y axes (tick labels)
plt.xticks(fontsize=12)  # X-axis tick label size
plt.yticks(fontsize=12)  # Y-axis tick label size
plt.legend(fontsize=15)
plt.tight_layout(h_pad=1)
















