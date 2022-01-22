'''
dual_eval.py

Solve the dual function optimally using basinhopping()

Jackie Zhong (jackie.z@wustl.edu)

Last edited: 12/18/21

'''

#reference: https://machinelearningmastery.com/bfgs-optimization-in-python/
#reference: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import Bounds

import matplotlib.pyplot as plt

# multiplier and dual values
q = []
m = []

def main():

    # maximize dual 

    # multiplier >= 0
    bounds = Bounds([0], [np.inf])
    minimizer_kwargs = {"method": "SLSQP", "bounds" : bounds}

    # outer loop, edit the first argument to test different function
    result_outer = basinhopping(example_function_5, [-1], niter=100, minimizer_kwargs=minimizer_kwargs) 

    # display result
    print(result_outer.x[0])
    print(-result_outer.fun)
    print(result_outer.nit)

    # plotting
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.autoscale()
    ax.plot(m, q)
    plt.show()


'''
    min x1 + x2

    s.t. g(x) = (x1 * x2) - 1 = 0

    x in R, x1 >= 0, x2 >= 0
'''
def example_function_1(multiplier):

    bounds = Bounds([0, 0], [np.inf, np.inf])

    minimizer_kwargs = {"method": "SLSQP", "bounds" : bounds}
    function = lambda x : (x[0] + x[1]) + multiplier[0] * (x[0]*x[1] - 1)
    inner_result = basinhopping(function, [0, 0], minimizer_kwargs=minimizer_kwargs)

    x_1 = inner_result.x[0]
    x_2 = inner_result.x[1]

    m.append(multiplier[0])
    q.append(x_1 + x_2 + multiplier[0] * (x_1 * x_2 - 1))

    return -(x_1 + x_2 + multiplier[0] * (x_1 * x_2 - 1))


'''
    min x

    s.t. g(x) = x^2 <=0

    x in R
'''
def example_function_2(mu):
    
    #start inner loop
    inner_result = basinhopping(lambda x : x[0] + mu[0]*x[0]**2, [0])

    # set x value from inner loop
    x_1 = inner_result.x[0]

    m.append(mu[0])
    q.append(x_1 + mu[0]*x_1**2)

    return -(x_1 + mu[0]*x_1**2)


'''
    min |x1| + x2

    s.t. g(x) = x1 <=0

    x in R, x2 >= 0
'''
def example_function_3(mu):

    bounds = Bounds([-np.inf, 0], [np.inf, np.inf])
    minimizer_kwargs = {"method": "SLSQP", "bounds" : bounds}
    inner_result = basinhopping(lambda x : abs(x[0]) + x[1] + mu[0]*(x[0]), [0, 0], minimizer_kwargs=minimizer_kwargs)

    x_1 = inner_result.x[0]
    x_2 = inner_result.x[1]

    m.append(mu[0])
    q.append(abs(x_1) + x_2 + mu[0]*x_1)

    return -(abs(x_1) + x_2 + mu[0]*x_1)


'''
    min x1 - x2

    s.t. g(x) = x1 + x2 - 1 <=0

    x in R, x1 >= 0, x2 >= 0
'''
def example_function_4(mu):

    #define bounds and start inner loop
    bounds = Bounds([0, 0], [np.inf, np.inf])
    minimizer_kwargs = {"method": "SLSQP", "bounds" : bounds}
    inner_result = basinhopping(lambda x : x[0] - x[1] + mu[0]*(x[0] + x[1] - 1), [0, 0], minimizer_kwargs=minimizer_kwargs)

    # set x value from inner loop
    x_1 = inner_result.x[0]
    x_2 = inner_result.x[1]

    m.append(mu[0])
    q.append(x_1 - x_2 + mu[0]*(x_1 + x_2 - 1))

    return -(x_1 - x_2 + mu[0]*(x_1 + x_2 - 1))
   

'''
    min 1/2(x1^* - x2*2)

    s.t. g(x) = x1 - 1 <=0

    x in R
'''
def example_function_5(mu):

    inner_result = basinhopping(lambda x : 1/2 * (x[0]**2 + x[1]**2) + mu[0]*(x[0] - 1), [0, 0])
    x_1 = inner_result.x[0]
    x_2 = inner_result.x[1]

    m.append(mu[0])
    q.append(1/2 * (x_1**2 + x_2**2) + mu[0]*(x_1 - 1))

    return -(1/2 * (x_1**2 + x_2**2) + mu[0]*(x_1 - 1))


if __name__ == "__main__":
    main()