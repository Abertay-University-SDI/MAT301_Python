#!/usr/bin/env python
# coding: utf-8

# # Numerical Integration
# Notebook to implement Euler, Midpoint and Euler Trapezium methods to solve differential equations. Designed to support Abertay University undergrad course MAT301 (JT, 2022).

# ## Motivation
# Many laws of motion (for example Newton's Laws) require differentiation or integration in order to extract different information. For example: integrating a displacement in time yields a velocity. The ability to integrate functions between two points is extremely useful in video games programming.
# 
# Given a function $f(x)$, we want to approximate the integral of $f(x)$ over the total **interval**, $[a,b]$. The following figure illustrates the area this generates. 
# 
# <img src="https://pythonnumericalmethods.berkeley.edu/_images/21.01.1-Illustration_integral.png" alt="Illustration integral" title="Illustration of the integral. The integral from a to b of the function f is the area below the curve (shaded in grey)." width="200"/>
# 
# To achieve this, we assume that the interval is broken up ("discretised"). Our independent variable, $x$, will comprise $N+1$ points with spacing, $h = \dfrac{b - a}{n}$. 
# 
# We denote each point in $x$ using the subscript "n", i.e. $x_n$ (which is common for this type of algorithm), beginning with 0 ($x_0 = a$) and ending with n ($x_n = b$). Note: There are $N+1$ grid points because the count starts at $x_0$. We also assume we have a function, $f(x)$, that can be computed for any of the grid points, or that we have been given the function implicitly as $f(x_n)$. The interval $[x_n, x_{n+1}]$ is referred to as a **subinterval**.
# 
# Several methods exist to approximate the area under the function or $\int_a^b f(x) dx$. Each method approximates the area under $f(x)$ for each subinterval by a shape for which it is easy to compute the exact area, and then sums the area contributions of every subinterval. In MAT301 we learned about Simpson's Rule and Trapezium Rule, which we will detail below.
# 

# #### Setting up Libraries
# 
# As always, we'll make use of some clever Python tools for plotting and maths, so our first step is to load in the libraries that store these tools:

# In[1]:


## Libraries
import numpy as np
import math 
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('{sys.executable} -m pip install termcolor')
from termcolor import colored


# 
# ## 1. Trapezium Rule
# The **Trapezium Rule** fits each subinterval with a trapezium and sums the areas of each trapezium to approximate the total integral. This approximation for the integral to an arbitrary function is shown in the following figure. 
# 
# <img src="https://pythonnumericalmethods.berkeley.edu/_images/21.03.1-Trapezoid_integral.png" alt="Trapezoid integral" title="Illustration of the trapezoid integral procedure. The area below the curve is approximated by a summation of trapezoids that approximate the function." width="400"/>
# 
# For each subinterval, the Trapezium Rule computes the area of a trapezium with corners at $(x_n, 0), (x_{n+1}, 0), (x_n, f(x_n))$, and $(x_{n+1}, f(x_{n+1}))$, which is $h\frac{f(x_n) + f(x_{n+1})}{2}$. Thus, the Trapezium Rule approximates integrals according to the expression
# 
# $$
# \int_a^b f(x)~{\rm d}x \approx \sum_{n=0}^{N-1} h\frac{f(x_n) + f(x_{n+1})}{2}.
# $$
# 
# 
# In the lectures you should have noted that the Trapezium Rule "double-counts" most of the terms in the series. To illustrate this fact, consider the expansion of the Trapezium Rule:
# 
# $$
# \begin{align}
# \sum_{i=0}^{N-1} h\left(\frac{f(x_n) + f(x_{n+1})}{2}\right) = &\frac{h}{2}\left[\left( f(x_0) + f(x_1)\right) + \left(f(x_1) + f(x_2)\right) + \right. \\
# &\left. \left(f(x_2) + f(x_3) \right) + \cdots + \left(f(x_{N-1}) + f(x_N)\right)\right].
# \end{align}
# $$
# 
# This results in lots of calls to $f(x)$ than is strictly necessary; we can be more computationally efficient using the following expression:
# 
# $$
# \int_a^b f(x)~{\rm d}x \approx \frac{h}{2} \left(f(x_0) + 2 \left(\sum_{i=1}^{N-1} f(x_n)\right) + f(x_N)\right).
# $$
# 

# In the lectures, we examined the Trapezium rule in Example 8.1 using the integral equation 
# 
# $$
# \int_0^1{\frac{\rm{d}x}{1+x^2}},
# $$
# where we were asked to use four strips and work to 5 decimal places.
# 
# To evaluate the problem, we broke up the range of integration into strips, evaluated the function at each of the edges of each strip, then applied the formula above to determine the total area under the curve. We'll do the same thing here numerically.
# 
# First, we'll define a Python function which returns the value of the function to be integrated at any given point in $x$: 

# In[2]:


def f1(x):
    result=1.0 / ( 1.0 + x * x )
    return result


# 
# Next, we'll define some useful parameters, like the upper and lower limit of integration and use these to determine the step size $h$ using the formula we learned in the lectures:

# In[3]:


a = 0.0
b = 1.0
N = 4
h = (b - a) / float(N)
print("step size h: ", h)


# To evaluate the integral, we need to evaluate the function at each step along the interval between $a$ and $b$. We therefore need to calculate the values of $x$ to pipe into our function. 
# 
# I will use a package function called *np.arange*, but this function returns all of the steps **except** the final step, hence when I call the function, I ask for values in the range from $a$ to $(b+h)$ so that the function returns values between $a$ and $b$ and omits a value I didn't want anyway! 

# In[4]:


x = np.arange(a, b + h, h)
print(x)


# Now that the $x$-range looks good, we can evaluate our function at each position, then carry out the sum according to the Trapezium Rule:

# In[5]:


f = f1(x)
total_trap_area = h * (0.5 * (f[0] + f[N]) + sum(f[1:N]))
print("total area (trapezium): ", total_trap_area)


# This was the value we calculated in the lectures by hand. Remember, we also identified that this integral had an exact solution: $\pi/4$.
# 
# The error in the Trapezium rule for this many strips and for this function can therefore be calculated by subtracting the approximation from the true value (using the absolute value in case the approximation yields an over-estimate):

# In[6]:


err_trap = abs(0.25 * np.pi - total_trap_area)
print("error in trapezium: ", err_trap)


# **Try for yourself:** Repeat the calculation above, but crank up the number of strips. What happens? 
# 
# (Careful not to ask for too many strips or the calculation speed may decrease!)

# ## Simpson's Rule
# 
# We also learned about a second approximate method to approximate finite integrals; Simpon's Rule.
# 
# 
# Consider *two* consecutive subintervals, $[x_{n-1}, x_n]$ and $[x_n, x_{n+1}]$. **Simpson's Rule** approximates the area under $f(x)$ over these two subintervals by fitting a quadratic polynomial through the points $(x_{n-1}, f(x_{n-1})), (x_n, f(x_n))$, and $(x_{n+1}, f(x_{n+1}))$, and then integrating the quadratic exactly. The following shows this integral approximation for an arbitrary function.
# 
# <img src="https://pythonnumericalmethods.berkeley.edu/_images/21.04.1-Simpson_integral.png" alt="Simpsons integral" title="Illustration of the Simpson integral formula. Discretization points are grouped by three, and a parabola is fit between the three points. This can be done by a typical interpolation polynomial. The area under the curve is approximated by the area under the parabola." width="550"/>
# 
# Quadratic polynomial approximation takes place over two sub-intervals. With some algebra and manipulation, and combining (integrating) various polynomials together, we arrive at the the formula
# 
# $$
# \int_a^b f(x)~{\rm d}x \approx \frac{h}{3} \left[f(x_0)+4 \left(\sum_{n=1, n\  {\text{odd}}}^{N-1}f(x_n)\right)+2 \left(\sum_{n=2, n\  {\text{even}}}^{N-2}f(x_n)\right)+f(x_N)\right].
# $$
# 
# **WARNING!** Remember that Simpson's Rule **must** be used for an even number of intervals and, therefore, an odd number of grid points.

# We can illustrate how Simpson's rule works, first by recycling the simple problem attacked earlier using the trapezium rule. We have already defined all the necessary components (limits, step size and values of $x$) that we will need. Let's first check that we're okay to apply Simpson's rule; to do this, we need to check if our total number of steps $n$ is even:

# In[7]:


if N % 2:
    print('WARNING: N EVEN - Simpsons rule will FAIL') # Odd 


# No warning appeared so we're good to go. All we need to do is sum the elements that we've calculated according to the Simpson's rule formula. We can be a bit clever here, and let Python decide which numbers are odd and which are even (by picking every second element in cleverly chosen ranges which omit $f_0$ and $f_n$ (used elsewhere in the formula.

# In[8]:


total_simp_area = h / 3.0 * ( f[0] + f[N] + 4.0 * sum(f[1:N:2]) + 2.0 * sum(f[2:N-1:2] ))
print("total area (Simpson's Rule):", total_simp_area)


# This number again looks very similar to the one derived by the Trapezium rule, but let's evaluate the error in it:

# In[9]:


err_simp = abs(0.25 * np.pi - total_simp_area)
print('error in Simpsons rule', err_simp)
print('error in Trapezium rule', err_trap)


# The error in the Simpson's rule implementation for this function (and number of strips) is much smaller than that of the Trapezium rule by around 3 orders of magnitude.

# ## Other methods
# 
# Python has an entire library of useful tools for these types of problems, including the ability to evaluate the trapezium method in a single line using *trapz*:

# In[10]:


from scipy.integrate import trapz
area_trapz = trapz(f,x)
print('TRAPZ: area=', area_trapz, ', error=', abs(0.25 * np.pi - area_trapz))


# Note that the error in the area from *trapz* is identical to that we found earlier creating this method manually. 
# 
# There are a few other numerical integration methods. Try *quad*:

# In[11]:


from scipy.integrate import quad
area_quad, err = quad(f1, 0, 1)

print('QUAD: area=', area_quad, ', error=', abs(0.25 * np.pi - area_quad))


# Now that you've seen one example of an integral performed using numerical techniques in Python, try some of the other worked examples in the lectures, or indeed some of the tutorial question **once you have done them by hand first**.

# In[ ]:




