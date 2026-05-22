#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
from scipy import optimize
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.optimize import fsolve

from utils import (poisson_regression,
                   negative_binomial_regression)


#########################################
### define the sample size and dimension
######################################### 

sample_size = 2000
# number_category = 5
# categorical_1 = np.random.choice(a= number_category, p = [0.5, 0.2, 0.1, 0.1, 0.1], size = sample_size)
# one_hot_categorical_1 = np.eye(number_category)[categorical_1]

dimension_numerical = 2

cov_x_poisson = np.ones(dimension_numerical)

def variance_sigma(variance_list = [1, 30, 50, 100]):
    for i in range(dimension_numerical):
        cov_x_poisson[i] = 1/(np.random.choice(variance_list)*sample_size)
    
    return cov_x_poisson

cov_x_poisson = variance_sigma(variance_list = [1,2,3,4,5])


x_numerical = np.random.multivariate_normal(mean=np.zeros(dimension_numerical), 
                                            # cov = np.identity(dimension_numerical), 
                                            cov = np.diag(cov_x_poisson),
                                            size = sample_size)

# x_raw = np.concatenate((x_numerical, one_hot_categorical_1), axis = 1)
x_raw = x_numerical

model = poisson_regression(x=x_raw, theta_loc = 0, theta_sigma = 1, bias = 1)
x_raw, y_raw = model.generate_data()
dimension = model.p

# model = negative_binomial_regression(x=x_raw, theta_loc = 0.5, theta_sigma = 1)
# x_raw, y_raw = model.generate_data()
# dimension = model.p


#################################
#### Read the real-world dataset
#################################

import pandas as pd
import numpy as np

df = pd.read_csv('german.csv', delimiter=';')

x_raw = df.drop("Creditability", axis=1)

# x_raw = df.drop(["Creditability", "Account_Balance"], axis=1)

y_raw = df.Creditability

from sklearn.preprocessing import StandardScaler

standizer = StandardScaler()
x_raw = standizer.fit_transform(x_raw)

y_raw = y_raw.to_numpy()

sample_size = x_raw.shape[0]

dimension_new = 20
x_numerical = np.random.multivariate_normal(mean=np.zeros(dimension_new), 
                                            cov = np.identity(dimension_new), 
                                            size = sample_size)

x_raw = np.concatenate((x_raw, x_numerical), axis = 1)
dimension = x_raw.shape[1]


from sklearn.preprocessing import StandardScaler

standizer = StandardScaler()
x_raw = standizer.fit_transform(x_raw)
pca = PCA(n_components=dimension)
x_raw=pca.fit_transform(x_raw)
y_raw = y_raw.to_numpy()

###########################################
##### Compute MLE and required ingredients
###########################################

####### Numerically solve the gradients to find MLE

def poisson_gradient(theta, x=x_raw, y = y_raw):
    gradient = 0
    for i in range(x.shape[0]):
        gradient += (y[i]-np.exp(x[i].dot(theta)))*x[i]
    gradient /= x.shape[0]
    return gradient

def mle_finder(function):
    return optimize.root(function, [0]*dimension)['x']

# def mle_finder(function):
#     return optimize.root(function, model.true_theta)['x']

# mle = mle_finder(poisson_gradient)



########  Find a GLM to find the MLE

import statsmodels.api as sm

poisson_model = sm.GLM(y_raw, x_raw, family=sm.families.Poisson())

poisson_model_results = poisson_model.fit()

mle = poisson_model_results.params

# mle = model.true_theta



# L = 0
# for i in range(sample_size):
#     L += (y_raw[i]-np.exp(x_raw[i].dot(mle)))*x_raw[i].reshape(dimension, 1)
# L /= sample_size



J = 0
V = 0 

for i in range(sample_size):
    J += np.exp(x_raw[i].dot(mle))*(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
    V += (y_raw[i]-np.exp(x_raw[i].dot(mle)))**2*(x_raw[i].reshape(dimension, 
                                                    1)@x_raw[i].reshape(dimension, 1).T)
    

V /= sample_size

J /=sample_size

Linear_term = 0

for i in range(sample_size):
    Linear_term += (x_raw[i].dot(mle))*(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
Linear_term /= sample_size

A = 0
for i in range(sample_size):
    A += (x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
A /= sample_size

### full-matrix inverse
precondition_matirx_poisson = np.linalg.inv(J)
J_inverse = precondition_matirx_poisson



### diagonal-matrix inverse
# precondition_matirx_poisson = np.linalg.inv(np.diag(np.diag(J)))

sanwich_matrix = J_inverse @ V @ J_inverse
scaled_sanwich_matrix = sanwich_matrix/sample_size

# initial state from normal distribution
# initial_state = np.random.normal(0, 1, dimension)

# initial state from zeros
# initial_state = np.zeros(dimension)

# initial state from MLE
initial_state = mle

scaled_precondition_matirx_poisson = precondition_matirx_poisson/sample_size



def derivative_transfer_function_poisson(x):
    return np.exp(x)

def transfer_function(x):
    return np.exp(x)
predicted_y = np.exp(x_raw @ mle)
errors = y_raw-predicted_y
E_vector = (errors.T @ x_raw)/sample_size
E = E_vector.reshape(dimension, 1) @ E_vector.reshape(1, dimension)

derivetives = []
for i in range(sample_size):
    derivetives.append(derivative_transfer_function_poisson((x_raw[i,:].T)@mle))
    
Derivative_vector = 0
for i in range(sample_size):
    Derivative_vector += derivetives[i]*(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)

Derivative_vector /= sample_size

def calculate_c1(cov):
    res = 0
    for i in range(sample_size):
        res += derivetives[i]**2*(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)@cov@(x_raw[i].reshape(dimension, 1)@x_raw[i].reshape(dimension, 1).T)
    
    res /= sample_size
    res -=  Derivative_vector@cov@Derivative_vector
    return res


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

def fluction_equation(Lambda, batch_size, cov_flatten):
    cov = cov_flatten.reshape(dimension, dimension)
    C = stationary_cov_noise(batch_size = batch_size, cov = cov)
    equation = Lambda@J@cov + cov@J@Lambda - Lambda@C@Lambda-Lambda@J@cov@J@Lambda
    return equation.flatten()


def fluction_equation_cts(Lambda, batch_size, cov_flatten):
    cov = cov_flatten.reshape(dimension, dimension)
    C = cov/batch_size
    equation = J@cov + cov@J - Lambda@C
    return equation.flatten()

def fluction_equation_discrete(Lambda, batch_size, cov_flatten):
    cov = cov_flatten.reshape(dimension, dimension)
    C = J/batch_size
    equation = Lambda@J@cov + cov@J@Lambda - Lambda@C@Lambda-Lambda@J@cov@J@Lambda
    return equation.flatten()


def calculate_cov_discrete(Lambda, batch_size):
    matrix = Lambda@J@(2*np.identity(dimension) - Lambda@J)
    C = J/batch_size
    return np.linalg.inv(matrix)@Lambda@C@Lambda
 

#########################################
### SGD framework for Poisson regression
#########################################
   

def sgd(n, batch_size, theta_0, lr_0, pre_matirx,
        fixed_lr=True,
         N = sample_size, precondition = False, fixed_point = False, 
         w1=1, w2=1, approximate_loss = True):
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
            if not approximate_loss:
                if not fixed_point:
                    delta_theta += (y_subsample[j]-np.exp(x_j.T @ pre_theta))*x_j/batch_size
                else:
                    delta_theta += (np.exp(x_j.T @ mle)-np.exp(x_j.T @ pre_theta))*x_j/batch_size
            else:
                delta_theta += np.exp(x_j.T @ mle)*(x_j@x_j.T)@(pre_theta - mle)

        delta_theta *= (gamma)
        delta_theta = precondition_matirx@delta_theta

        
        ret[i+1, :] = pre_theta+delta_theta.reshape(dimension,)
    return ret

####################################
### sample with fixed learning rate
####################################

b = 32
lr_list = [0.1]

# lr_list = [5, 10, 15]
### set initilization

initial_state = mle
# initial_state = np.random.normal(0, 1, dimension)


num_epochs = 1000


samples_record_sgd_path = {}
theoretical_cov_path = {}
countinuous_cov_path = {}
discrete_cov_path = {}
empirical_cov_path = {}


for l in range(len(lr_list)):
    # index = [x*n_iterations[l] for x in range(number_epochs+1)]
    num_iterations = num_epochs *int(sample_size/b)
    
    samples_path_sgd = sgd(n=num_iterations, batch_size = b, 
                           pre_matirx = np.identity(dimension),
                             theta_0=initial_state, lr_0 = lr_list[l], fixed_lr=True, w1=1, w2=1,
                             fixed_point=False,
                             precondition=True, approximate_loss = False)

    samples_record_sgd_path[l] = samples_path_sgd
    
    # Initial guess (flattened version of a dxd matrix)
    initial_guess = np.identity(dimension).flatten()
    # initial_guess = np.zeros((dimension, dimension)).flatten()


    # Solve the system
    solution_flat = fsolve(lambda x: fluction_equation(Lambda = lr_list[l]*np.identity(dimension), 
                                                       batch_size = b, cov_flatten = x), initial_guess)
        
    theoretical_cov = solution_flat.reshape(dimension, dimension)
    
    cts_cov = lr_list[l]/b*np.identity(dimension)
    
    # discrete_solution_flat = fsolve(lambda x: fluction_equation_discrete(Lambda = lr_list[l]*np.identity(dimension), 
    #                                                    batch_size = b, cov_flatten = x), initial_guess)
    # discrete_cov = discrete_solution_flat.reshape(dimension, dimension)
    discrete_cov = calculate_cov_discrete(Lambda = lr_list[l]*np.identity(dimension), batch_size = b)
    
    theoretical_cov_path[l] = theoretical_cov
    countinuous_cov_path[l] = cts_cov
    discrete_cov_path[l] = discrete_cov
    empirical_cov_path[l] = np.cov(samples_path_sgd.T)


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

###################################
### make single plot for single lr
###################################
# Plot settings
fig, ax = plt.subplots(figsize = (5,5))

index = 0

# n_samples_for_plots = samples_record_sgd_path[l].shape[0]
n_samples_for_plots = 3000


for index in range(len(lr_list)): 

    cov_true = [[theoretical_cov_path[index][index_1, index_1], theoretical_cov_path[index][index_1, index_2]],
                [theoretical_cov_path[index][index_2, index_1], theoretical_cov_path[index][index_2, index_2]]]
    
    cov_cts = [[countinuous_cov_path[index][index_1, index_1], countinuous_cov_path[index][index_1, index_2]],
                [countinuous_cov_path[index][index_2, index_1], countinuous_cov_path[index][index_2, index_2]]]
    
    cov_discrete = [[discrete_cov_path[index][index_1, index_1], discrete_cov_path[index][index_1, index_2]],
                [discrete_cov_path[index][index_2, index_1], discrete_cov_path[index][index_2, index_2]]]
    
    
    # x_samples = samples_record_sgd_path[index][-n_samples_for_plots:, index_1]
    # y_samples = samples_record_sgd_path[index][-n_samples_for_plots:, index_2]
    samples=np.random.multivariate_normal(mean = mle, cov=empirical_cov_path[index], size=n_samples_for_plots)
    
    x_samples = samples[:, index_1]
    y_samples = samples[:, index_2]
    
    
    # Create a figure with 3 subplots in a row (1 row, 3 columns)

    # ax.axhline(0, color='grey', lw=1)
    # ax.axvline(0, color='grey', lw=1)
    
    # Plotting the 3-sigma ellipse
    
    ax.scatter(x_samples, y_samples, marker = '*', 
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
    x_min = min(x_extents_cts[0], x_extents_discrete[0], x_extents_true[0])-1.5
    x_max = max(x_extents_cts[1], x_extents_discrete[1], x_extents_true[1])+1.5
    y_min = min(y_extents_cts[0], y_extents_discrete[0], y_extents_true[0])-1.5
    y_max = max(y_extents_cts[1], y_extents_discrete[1], y_extents_true[1])+1.5
    
    # Set the individual axis limits for this subplot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
        
    
    
    custom_lines = [
                    Line2D([0], [0], color='black', lw=2, linestyle = '-'),
                    Line2D([0], [0], color='#1f77b4', lw=2, linestyle = '--'),
                    Line2D([0], [0], color='#d62728', lw=2, linestyle = '-.'),
                    Line2D([0], [0], marker='*', color='#bcbd22', lw=0, markersize=6)  # Point marker
                    ]
    # if index==0:
        # Add a legend with straight lines instead of ellipses
        # axes[index].legend(custom_lines, ['Continuous-time', 'discrete-quadratic+constant noise', 'discrete-quadratic+exact noise(this paper)', 'Iterates'], 
        #                     fontsize=13)
        # Add the legend outside the plot (right side of the figure)
        # Remove x and y axis ticks (numbers)
        # axes[index].set_xticks([])
        # axes[index].set_yticks([])
    # ax.legend(custom_lines, ['Continuous-time', 'discrete-quadratic+constant noise', 'discrete-quadratic\nexact noise(this paper)', 'Iterates'], 
    #                     fontsize=11)
    # ax.set_title(r'$\lambda$ ={}'.format(lr_list[index]), fontsize=16)
    fig.legend(custom_lines, 
                ['continuous-time', 'discrete-quadratic+constant noise', 'discrete-quadratic\nexact noise(this paper)', 'Iterates'], 
                loc='center right', bbox_to_anchor=(0.99, 0.8), fontsize=13)
# plt.show()
fig.tight_layout(h_pad=1)




#########################################
### make multiple plots for multiple lrs
########################################


# n_samples_for_plots = samples_record_sgd_path[l].shape[0]
n_samples_for_plots = 1000

# Create a figure with 3 subplots in a row (1 row, 3 columns)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for index in range(len(lr_list)): 

    cov_true = [[theoretical_cov_path[index][index_1, index_1], theoretical_cov_path[index][index_1, index_2]],
                [theoretical_cov_path[index][index_2, index_1], theoretical_cov_path[index][index_2, index_2]]]
    
    cov_cts = [[countinuous_cov_path[index][index_1, index_1], countinuous_cov_path[index][index_1, index_2]],
                [countinuous_cov_path[index][index_2, index_1], countinuous_cov_path[index][index_2, index_2]]]
    
    cov_discrete = [[discrete_cov_path[index][index_1, index_1], discrete_cov_path[index][index_1, index_2]],
                [discrete_cov_path[index][index_2, index_1], discrete_cov_path[index][index_2, index_2]]]
    
    
    # x_samples = samples_record_sgd_path[index][-n_samples_for_plots:, index_1]
    # y_samples = samples_record_sgd_path[index][-n_samples_for_plots:, index_2]
    samples=np.random.multivariate_normal(mean = mle, cov=empirical_cov_path[index], size=n_samples_for_plots)
    
    x_samples = samples[:, index_1]
    y_samples = samples[:, index_2]
    
    
    # Create a figure with 3 subplots in a row (1 row, 3 columns)

    # ax.axhline(0, color='grey', lw=1)
    # ax.axvline(0, color='grey', lw=1)
    
    # Plotting the 3-sigma ellipse
    
    axes[index].plot(x_samples, y_samples, '*', 
                alpha=0.01, color='#bcbd22', markersize=110)
    
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
                    Line2D([0], [0], color='#1f77b4', lw=2, linestyle = '--'),
                    Line2D([0], [0], color='#d62728', lw=2, linestyle = '-.'),
                    Line2D([0], [0], marker='*', color='#bcbd22', lw=0, markersize=6)  # Point marker
                    ]
    # if index==0:
        # Add a legend with straight lines instead of ellipses
        # axes[index].legend(custom_lines, ['Continuous-time', 'discrete-quadratic+constant noise', 'discrete-quadratic+exact noise(this paper)', 'Iterates'], 
        #                     fontsize=13)
        # Add the legend outside the plot (right side of the figure)
        # Remove x and y axis ticks (numbers)
        # axes[index].set_xticks([])
        # axes[index].set_yticks([])
    axes[0].legend(custom_lines, ['Continuous-time', 'discrete-quadratic+constant noise', 'discrete-quadratic\nexact noise(this paper)', 'Iterates'], 
                        fontsize=11)
    axes[index].set_title(r'$\lambda$ ={}'.format(lr_list[index]), fontsize=16)
    # fig.legend(custom_lines, 
    #            ['continuous-time', 'discrete-quadratic\nconstant noise', 'discrete-quadratic\nexact noise(this paper)', 'Iterates'], 
    #            loc='center right', bbox_to_anchor=(1.08, 0.75), fontsize=13)
# plt.show()
fig.tight_layout(h_pad=1)












