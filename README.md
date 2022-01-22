# optimization-dual-function-solver

Emperically evaluate as well as plot a dual function subjected to optimization. See daul_func.pdf for more background.


# Usage

To use different test cases, edit line 19 in dual_eval.py:

result_outer = basinhopping(<name of the function>>, [1], niter=100, minimizer_kwargs=minimizer_kwargs) 

e.g. result_outer = basinhopping(example function, [1], niter=100, minimizer_kwargs=minimizer_kwargs) 
  
See example functions in the code on how to represent an problem with a function. 

For further referecne of basinhopping, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

Dependencies: numpy, scipy
  
To run the program, issue:

python dual_eval.py
