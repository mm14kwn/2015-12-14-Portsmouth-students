#!/usr/bin/env python
import numpy as np;
from scipy import optimize, special, linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from scipy.integrate import odeint;

def x1_t(t,x1,x2,k,m): # exact solution for mass 1 at time t
    w=np.sqrt(k/m);    # initial velocity assumed zero
    a1=(x1+x2)/2.0;
    a2=(x1-x2)/2.0;
    return a1*np.cos(w*t) + a2*np.cos(np.sqrt(3)*w*t);

def x2_t(t,x1,x2,k,m): # exact solution for mass 2 at time t
    w=np.sqrt(k/m);    # initial velocity assumed zero
    a1=(x1+x2)/2.0;
    a2=(x1-x2)/2.0;
    return a1*np.cos(w*t) - a2*np.cos(np.sqrt(3)*w*t);

def vectorfield(w, t, p):
    """Defines differential equations for the coupled masses
        w :  vector of the state variables: w = [x1,v1,x2,v2]
        t :  time
        p :  vector of the parameters: p = [m,k] """
    x1, v1, x2, v2 = w;
    m, k = p;

    # Create f = (x1',y1',x2',y2'):
    f = [v1, (-k * x1 + k * (x2 - x1)) / m, 
         v2, (-k * x2 - k * (x2 - x1)) / m];
    return f;

x1_sol = np.vectorize(x1_t);
x2_sol = np.vectorize(x2_t);


# Parameters and initial values
m = 1.0; k = 1.0;     # mass m, spring constant k
x01 = 0.5; x02 = 0.0; # Initial displacements
v01 = 0.0; v02 = 0.0; # Initial velocities : LEAVE AS ZERO 

# ODE solver parameters
abserr = 1.0e-8; relerr = 1.0e-6; 
stoptime = 10.0; numpoints = 250;


# Create time samples for the output of the ODE solver
t = np.linspace(0, stoptime, numpoints);

# Pack up the parameters and initial conditions as lists/arrays:
p =  [m, k];  w0 = [x01, v01, x02, v02];

# Call the ODE solver. Note: args is a tuple
wsol = odeint(vectorfield, w0, t, args=(p,), atol=abserr, 
              rtol=relerr);

# print x1_sol                   
# fig=plt.figure()
# plt.plot(x1_sol)
# fig.savefig('x1.pdf')


# Print and save the solution
with open('coupled_masses.dat', 'w') as f: 
    for t1, w1 in zip(t, wsol):
        print >> f, t1, w1[0], w1[1], w1[2], w1[3]


import numpy as np
a=np.array([[0,1,2],[3,4,5]]); b=np.array([[-5,-6,-7],[-8,-9,-10]])
print np.dstack((a,b)).shape
print a.shape


# import modules for plotting
import matplotlib.pyplot as plt;
from matplotlib.font_manager import FontProperties;

# get saved values from saved file
t, x1, v1, x2, v2 = np.loadtxt('coupled_masses.dat', unpack=True);


# figure proporties
plt.figure(1, figsize=(10, 3.5)); plt.xlabel('t'); 
plt.ylabel('x'); plt.grid(True); plt.hold(True); 

# plot exact solutions
time=np.linspace(0,stoptime,50);

plt.plot(time, x1_sol(time,x01,x02,k,m), 'r*', linewidth=1);
plt.plot(time, x2_sol(time,x01,x02,k,m), 'mo', linewidth=1);

# plot numerical solutions
plt.plot(t, x1, 'b-', linewidth=1); plt.plot(t, x2, 'g-', linewidth=1);

plt.legend(('$x_{1,sol}$', '$x_{2,sol}$', '$x_{1,num}$', 
        '$x_{2,num}$'),prop=FontProperties(size=12));
plt.title('Mass Displacements for the\nCoupled Spring-Mass System');
plt.savefig('coupled_masses.png', dpi=100); # save figure

