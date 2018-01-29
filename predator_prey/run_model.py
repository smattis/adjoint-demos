import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

'''
Predator-Prey model:  \dot{X}=\begin{array}{ll}{ \alpha & \beta x_0 \\ \tau x_1 & \lambda \right] X

 params: Initial Conditions: X1[0], X2[0] and Coefficients: alpha, beta, tau, gamma

'''


def pred_res(Xcur,Xold,delt,alpha,beta,tau,gamma):
    res=Xcur-Xold-delt*np.array([alpha*Xcur[0]-beta*Xcur[0]*Xcur[1], tau*Xcur[0]*Xcur[1]-gamma*Xcur[1]])
    print 'Current Residual Norm: %f'%(linalg.norm(res))
    return res

def pred_jac(X,alpha,beta,tau,gamma):
    jac = np.array([[alpha-beta*X[1],-beta*X[0]],[ tau*X[1],tau*X[0]-gamma]])
    return jac

def lb_model0(input_data, plot=False):
    return run_model_master(input_data, delt=0.25, plot=plot)

def lb_model1(input_data, plot=False):
    return run_model_master(input_data, delt=0.1, plot=plot)

def lb_model2(input_data, plot=False):
    return run_model_master(input_data, delt=0.01, plot=plot)

def lb_model3(input_data, plot=False):
    return run_model_master(input_data, delt=0.001, plot=plot)

def run_model_master(input_data, delt, plot=False):
    """
    Solve model.
    
    input_data: parameter sets (num_samples x 6) with columns
X1(0), X2(0), alpha, beta, tau, gamma
    delt: time step size

    output: (Y1, Y2, Y3)
    Y1: X1(5), X2(5), X1(10), and X2(1) (num_samples x 4)
    Y2: error estimates (num_samples x 4)
    Y3: Jacobians (num_samples x 4 x 6)

    """
    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    num_qoi = 4
    
    values = np.zeros((num_runs, num_qoi))
    jacobians = np.zeros((num_runs, num_qoi, num_runs_dim))
    error_estimates = np.zeros((num_runs, num_qoi))

    for k in range(num_runs):
        # time mesh points and \delta t
        time = np.arange(0.0, 10.0+delt, delt)
        
        # solution matrix X(2xtimesteps) for storing solution.
        X = np.zeros((2,len(time)))
        # Initial condition X(:,1)
        P1 = input_data[k, 0]
        P2 = input_data[k, 1]
        X[:,0]=np.array([P1,P2])
        
        # Model parameters
        alpha = input_data[k, 2]
        beta = input_data[k, 3]
        tau = input_data[k, 4]
        gamma = input_data[k, 5]
        
        # tolerance for newton solve
        tol = 1E-10
        
        # Dorward timestepping Solve: Backward Euler (DG0)
        Xcur = X[:,0]
        for tstep in range(1,len(time)):
            Xold = Xcur
            # Newton loop
            res = pred_res(Xcur,Xold,delt,alpha,beta,tau,gamma)
            print 'Time Step: %d'%(tstep)
            while np.max(np.abs(res)) > tol:
                # compute Newton step
                jac = pred_jac(Xcur,alpha,beta,tau,gamma)
                newton = linalg.solve(np.eye(2)-delt*jac,-res)
                print 'Norm step: %f'%(linalg.norm(newton))
                Xcur = Xcur+newton
                res = pred_res(Xcur,Xold,delt,alpha,beta,tau,gamma)
                
            # update and store X at new time value
            X[:,tstep] = Xcur

        # Calculate QoI
        lt = len(time)-1
        lts = [lt/2, lt]
        
        values[k,0] = X[0, lts[0]]
        values[k,1] = X[1, lts[0]]
        values[k,2] = X[0, lt]
        values[k,3] = X[1, lt]

        for i in range(len(lts)):
            # Adjoint (Reverse) Solve:Crank-Nicholson(CG1) for Prey
            Phi = np.zeros((2, lts[i]+1))
            # Adjoint Initial condition: QoI is Prey population at t=T(final)
            Phi[:,-1]=np.array([1.0 ,0.0])
            for tstep in range(lts[i]-1,-1,-1):
                # Get Jacobian values at current time interval and assemble RHS
                RHS=Phi[:,tstep+1]+.5*delt*np.dot(pred_jac(X[:,tstep+1],alpha,beta,tau,gamma).transpose(),Phi[:,tstep+1])
                # Get Jacobian values at new time and build matrix
                AMAT=np.eye(2)-delt*.5*pred_jac(X[:,tstep],alpha,beta,tau,gamma).transpose()
                # Solve for new time value
                Phi[:,tstep]=linalg.solve(AMAT,RHS)

            # Error Contributions:
            # No Initial Error (X(0),Phi(0))

            # X is piecewise constant on a time interval and Phi is piecewise linear on a time interval
            # Residual contribution on time interval with midpoint quadrature (-F(X),\Phi)_{T_i}

            # Jump Error from DG=(X(t-)-X(t+),\Phi(2))
            step_err=np.zeros(time.shape)
            jump_err=np.zeros(time.shape)
            step_err[0]=0.0
            jump_err[0]=0.0
            for tstep in range(1,lts[i]+1):
               tres=[alpha*X[0,tstep]-beta*X[0,tstep]*X[1,tstep], tau*X[0,tstep]*X[1,tstep]-gamma*X[1,tstep]]
               Phimid=.5*(Phi[:,tstep]+Phi[:,tstep-1]) # \Phi at midpoint
               step_err[tstep]=delt*np.dot(tres,Phimid)#.transpose()
               if tstep<len(time):
                  jump_err[tstep]=np.dot(X[:,tstep-1]-X[:,tstep],Phi[:,tstep])
            error_estimates[k, i*2] = np.sum(jump_err) + np.sum(step_err)

            # Calculate derivatives
            # Derivatives wrt initial conditions
            jacobians[k,i*2,0] = Phi[0,0]
            jacobians[k,i*2,1] = Phi[1,0]

            # Parameter Derivatives
            d_a = 0.0
            d_b = 0.0
            d_t = 0.0
            d_g = 0.0
            for tstep in range(1,lts[i]+1):
               tres_a = [X[0,tstep], 0.0]
               tres_b = [-X[0,tstep]*X[1,tstep], 0.0]
               tres_t = [0.0, X[0,tstep]*X[1,tstep]]
               tres_g = [0.0, -X[1,tstep]]
               Phimid=.5*(Phi[:,tstep]+Phi[:,tstep-1]) # \Phi at midpoint
               d_a += delt*np.dot(tres_a, Phimid)
               d_b += delt*np.dot(tres_b, Phimid)
               d_t += delt*np.dot(tres_t, Phimid)
               d_g += delt*np.dot(tres_g, Phimid)
            jacobians[k, i*2, 2] = d_a
            jacobians[k, i*2, 3] = d_b
            jacobians[k, i*2, 4] = d_t
            jacobians[k, i*2, 5] = d_g

            # Adjoint (Reverse) Solve:Crank-Nicholson(CG1) for Predator
            Phi = np.zeros((2, lts[i]+1))
            # Adjoint Initial condition: QoI is Prey population at t=T(final)
            Phi[:,-1]=np.array([0.0 ,1.0])
            for tstep in range(lts[i]-1,-1,-1):
                # Get Jacobian values at current time interval and assemble RHS
                RHS=Phi[:,tstep+1]+.5*delt*np.dot(pred_jac(X[:,tstep+1],alpha,beta,tau,gamma).transpose(),Phi[:,tstep+1])
                # Get Jacobian values at new time and build matrix
                AMAT=np.eye(2)-delt*.5*pred_jac(X[:,tstep],alpha,beta,tau,gamma).transpose()
                # Solve for new time value
                Phi[:,tstep]=linalg.solve(AMAT,RHS)

            # Error Contributions:
            # No Initial Error (X(0),Phi(0))

            # X is piecewise constant on a time interval and Phi is piecewise linear on a time interval
            # Residual contribution on time interval with midpoint quadrature (-F(X),\Phi)_{T_i}

            # Jump Error from DG=(X(t-)-X(t+),\Phi(2))
            step_err=np.zeros(time.shape)
            jump_err=np.zeros(time.shape)
            step_err[0]=0.0
            jump_err[0]=0.0
            for tstep in range(1,lts[i]+1):
               tres=[alpha*X[0,tstep]-beta*X[0,tstep]*X[1,tstep], tau*X[0,tstep]*X[1,tstep]-gamma*X[1,tstep]]
               Phimid=.5*(Phi[:,tstep]+Phi[:,tstep-1]) # \Phi at midpoint
               step_err[tstep]=delt*np.dot(tres,Phimid)#.transpose()
               if tstep<len(time):
                  jump_err[tstep]=np.dot(X[:,tstep-1]-X[:,tstep],Phi[:,tstep])
            error_estimates[k, i*2+1] = np.sum(jump_err) + np.sum(step_err)

            # Calculate derivatives
            # Derivatives wrt initial conditions
            jacobians[k,i*2+1,0] = Phi[0,0]
            jacobians[k,i*2+1,1] = Phi[1,0]

            # Parameter Derivatives
            d_a = 0.0
            d_b = 0.0
            d_t = 0.0
            d_g = 0.0
            for tstep in range(1,lts[i]+1):
               tres_a = [X[0,tstep], 0.0]
               tres_b = [-X[0,tstep]*X[1,tstep], 0.0]
               tres_t = [0.0, X[0,tstep]*X[1,tstep]]
               tres_g = [0.0, -X[1,tstep]]
               Phimid=.5*(Phi[:,tstep]+Phi[:,tstep-1]) # \Phi at midpoint
               d_a += delt*np.dot(tres_a, Phimid)
               d_b += delt*np.dot(tres_b, Phimid)
               d_t += delt*np.dot(tres_t, Phimid)
               d_g += delt*np.dot(tres_g, Phimid)
            jacobians[k, i*2+1, 2] = d_a
            jacobians[k, i*2+1, 3] = d_b
            jacobians[k, i*2+1, 4] = d_t
            jacobians[k, i*2+1, 5] = d_g
        if plot:
            plt.figure()
            plt.plot(range(0,len(time)), X[0,:])
            plt.plot(range(0,len(time)), X[1,:])
            plt.xlabel("t")
            plt.legend(['X1', 'X2'])
            plt.show()


    return (values, -error_estimates, jacobians)
                    
                      
            
