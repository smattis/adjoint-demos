import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# Python script to generate discrete error estimate at \f$u(x=.5)\f$ for the 2 point BVP
# \f$u^\prime\prime=f(x)=e^{\alpha x)}\f$, \f$u(0)=u(1)=0\f$, solved using a three-point CFD stencil,
# with a uniform grid of size \f$h=.05\f$.

# \f$\alpha=0\f$ so RHS = 1;
# Setup Computational grid on \f$[h,1-h]\f$.
alpha=0
h=.05
xval = np.arange(h, 1.0, h)
pts=len(xval)
# Setup Forward Solution:

# Discretize PDE: \f[ -u^\prime\prime(x_n)= \frac{2u(x_n)-u(x_{n-1})-u(x_{n+1})}{h^2} = e^{\alpha x_n} \f]

# Uniform grid so grid size h can be moved to RHS
b = h**2*np.exp(alpha*xval)
# We use the spdiags command to map -1 2 1 to the tridiagonal matrix A
temp = np.hstack((-np.ones((pts,1)), 2.0*np.ones((pts,1)), -np.ones((pts,1)))).transpose()
A = sparse.spdiags(temp, [-1,0,1], pts, pts, format = "csr")

# Create our approximate solution u_sol by 7 steps of CG(no preconditioner)
(u_sol,_) = splinalg.cg(A, b,tol=1.0e-20, maxiter=7)
# Solve for "truth" solve.
u = splinalg.spsolve(A,b)

# Now set up the adjoint problem and solve it("exactly").
# QoI is u_sol(10), so \f$ \psi = e_{10}\f$.
psi = np.zeros((pts,1))
psi[9] = 1

# Atop=A'; Atop=A^\top not needed as A is symetric
# Solve for adjoint solution using slash for "truth"
phi = splinalg.spsolve(A,psi)

# Compute residual vector
Res=b-A.dot(u_sol)
# Estimated Error = \f$(R(U),\phi)\f$=(b-AU,\phi)=(b-AU)^\top\phi\f$.
err_est=np.dot(Res,phi)
print "Error estimate for QoI 1: ", err_est

# Real Error u(10)-u_sol(10)
err=u[9]-u_sol[9]
print "True Error for QoI 1: ", err

# Effectivity Index is Est/Err/
eff=err_est/err
print "Effectivity ratio: ", eff

# second QoI=Average value of u on [.6,.8]
psi2 = np.zeros((pts,1))
psi2[11:16] = 0.2

# Adjoint matrix is the same
phi2 = splinalg.spsolve(A,psi2)

# Repeat Error block
err_est2=np.dot(Res,phi2)
print "Error estimate for QoI 2: ", err_est2
err2=np.mean(u[11:16]-u_sol[11:16])
print "True Error for QoI 2: ", err2
eff2 = err_est2/err2;
print "Effectivity ratio: ", eff2

# Some plotting and discussion

plt.figure(0)
plt.plot(xval, u_sol, 'b*', xval, u, 'r-')
plt.legend(['U approx','U'])
plt.savefig('forward_solution.png')

# Influence functions: Adjoint solutions
plt.figure(1)
plt.plot(xval, phi, xval, phi2)
plt.legend([r'$\phi_{.5}$',r'$\phi_{avr}$'])
plt.savefig('adjoint_solution.png')

# "Local Error Contributions"
plt.figure(2)
plt.plot(xval, u-u_sol, xval, Res*phi, xval, Res*phi2)
plt.legend(['error in U', 'error id 1', 'error id 2'])
plt.savefig('local_error.png')

plt.show()
