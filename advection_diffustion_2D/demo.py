from dolfin import *
import numpy as np
import math

# Solves the steady linear convection-diffusion problem.
# Setup Mesh
ny = 50
nx = 5*ny
mymesh = RectangleMesh(Point(0.0, 0.0),Point(10.0,2.0), nx, ny)

f = Constant((1.0)) # RHS
# Define convection and diffusion
diffusion = Expression("0.05+tanh(10.0*pow(x[0]-5.0,2) + 10.0*pow(x[1]-1,2))", degree=2)
convection = Constant((-100.0, 0.0))

# Define QoI as average value over box
boxX = [1.0, 3.0]
boxY = [0.5, 1.5]
class MyExpression0(Expression):
    def __init__(self,boxX,boxY, **kwargs):
        self.boxX=boxX
        self.boxY=boxY
        self.area = (boxX[1]-boxX[0])*(boxY[1]-boxY[0])

    def eval(self,value, x): 
        if x[0] >= self.boxX[0] and x[0] <= self.boxX[1] and x[1] >= self.boxY[0] and x[1] <= self.boxY[1]:
            value[0] = 1.0/self.area
        else:
            value[0] = 0.0
psi = MyExpression0(boxX=boxX, boxY=boxY, degree=2) 

# Define forward problem
V1 = FunctionSpace(mymesh, 'Lagrange', 1)

# Define Boundary Subdomains and BCS
class RightBoundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return x[0] > (10.0 - 5.0*DOLFIN_EPS) and on_boundary

class TBBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > DOLFIN_EPS and x[0] < (10.0 - DOLFIN_EPS) and on_boundary

RightBC = Expression("x[1]*(2.0-x[1])", degree=2)
ZeroBC = Constant(0.0)

bcr = DirichletBC(V1, RightBC, RightBoundary())
bctb = DirichletBC(V1, ZeroBC, TBBoundary())

u1 = TrialFunction(V1)
v1 = TestFunction(V1)

a1 = (dot(convection, grad(u1))*v1 + diffusion*inner(nabla_grad(u1), nabla_grad(v1)))*dx
L1 = f*v1*dx

# Solve Forward problem
u1 = Function(V1)
solve(a1 == L1, u1, [bcr,bctb])

# Calculate QoI
q = assemble(dot(u1,psi)*dx)
print "Computed QoI: ", q

# Save Solution
file = File("plots/forward.pvd")
u1.rename("u1", "Forward Solution")
file << u1
# Plot solution
#plot(u1, interactive=False, title='forward solution')

# Define adjoint problem 
V2 = FunctionSpace(mymesh, 'Lagrange', 2)
u2 = TrialFunction(V2)
v2 = TestFunction(V2)

a2 = (-div(convection*u2)*v2 + diffusion*inner(nabla_grad(u2), nabla_grad(v2)))*dx
L2 = psi*v2*dx
bcr = DirichletBC(V2, ZeroBC, RightBoundary())
bctb = DirichletBC(V2, ZeroBC, TBBoundary())

# Solve adjoint problem 
u2 = Function(V2)
solve(a2 == L2, u2, [bcr, bctb])

# Save adjoint solution
file = File("plots/adjoint.pvd")
u2.rename("u2", "Adjoint Solution")
file << u2
# Plot adjoint solution
#plot(u2, interactive=False, title = 'adjoint solution')

# Error identifier
f = interpolate(f,V1)
errorID = (f*u2 - dot(convection, grad(u1))*u2 - diffusion*inner(nabla_grad(u1), nabla_grad(u2)))
errorEstimate = assemble(errorID*dx)
print "Error estimate: ", errorEstimate
#plot(errorID, interactive=True, title = 'error identifier')
file = File("plots/errorID.pvd")
errorID = project(errorID, V1)
errorID.rename("errorID", "error identifier")
file << errorID

