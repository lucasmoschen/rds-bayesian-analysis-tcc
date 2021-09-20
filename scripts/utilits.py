#!usr/bin/env python 

import numpy as np
from scipy.optimize import minimize

class ParameterAlpha: 

    def __init__(self) -> None:
        pass

    def relative_quadratic_error(self,x,y): 
        return (x - y)**2/x**2

    def quadratic_error(self,x,y): 
        return (x - y)**2

    def loss_function(self, alpha, m1, m2, v1, v2, rho, g, c = [1,1,1,1]): 
        """
        This function calculates the loss function
        g(m1,E[X1])+g(m2,E[X2])+g(v1,Var[X1])+g(v2,Var[X2])+g(rho,Cor(X1,X2)) 
        as function of alpha, when the means (m1,m2) and variances (v1,v2)
        from sensitivity and specificity are fixed, and the correlation rho
        between them. 
        - alpha: vector[4]
        - m1,m2,v1,v2,rho: float between 0 and 1. 
        - g: a function that receives to real values and return a real value.
        - c: vector[4] represents the weights. Default: [1,1,1,1].
        """
        alpha_tilde = sum(alpha)
        div = alpha_tilde*alpha_tilde*(alpha_tilde + 1)
        
        alpha_12 = alpha[0] + alpha[1]
        alpha_34 = alpha[2] + alpha[3]
        alpha_13 = alpha[0] + alpha[2]
        alpha_24 = alpha[1] + alpha[3]
        
        obj  = c[0]*g(m1, alpha_12/alpha_tilde)
        obj += c[1]*g(m2, alpha_13/alpha_tilde)
        obj += c[2]*g(v1, alpha_12*alpha_34/div)
        obj += c[3]*g(v2, alpha_13*alpha_24/div)
        obj += g(rho, (alpha[0]*alpha[3] - alpha[1]*alpha[2])/(np.sqrt(alpha_12*alpha_34*alpha_13*alpha_24)))
        
        return obj

    def minimizer(self, m1, m2, v1, v2, rho, 
                  c = [1,1,1,1], 
                  g = 'quadratic', 
                  x0 = (1,1,1,1)): 
        """
        Minimized the loss function as function of alpha. 
        """
        if g == 'relative_quadratic': 
            g = self.relative_quadratic_error
        else: 
            g = self.quadratic_error      

        return minimize(fun = self.loss_function, 
                        x0 = x0, 
                        args = (m1, m2, v1, v2, rho, g, c),
                        bounds = [(0, np.inf)]*4,
                        method = 'trust-constr')

    def solve_equations(self, m1, m2, v1, v2, rho):
        """
        Let (X,Y) ~ BivariateBeta(a1,a2,a3,a4). This functions aim to solve
        the system m1 = E[X], m2 = E[Y], v1 = Var(X), rho = Cor(X,Y) nad
        subject to the induced error on v2. 
        """
        # Finds alpha4 as function of rho
        rho_expression = rho + np.sqrt((1-m1)*(1-m2)/(m1*m2))
        rho_expression *= np.sqrt(m1*m2*(1-m1)*(1-m2))*(m1 - m1*m1 - v1)
        alpha4 = rho_expression/v1

        if alpha4 < 0: 
            raise Exception('This system does not have a solution since alpha_4 < 0')
     
        # finds alpha1,alpha2,alpha3 as function of alpha4, m1, m2, and v1
        C = (m1 - m1*m1 - v1)/v1 
        alpha1 = (m1 + m2 - 1)*C + alpha4 
        alpha2 = (1 - m2)*C - alpha4 
        alpha3 = (1 - m1)*C - alpha4 

        if alpha1 < 0: 
            raise Exception('This system does not have a solution since alpha_1 < 0')
        elif alpha2 < 0: 
            raise Exception('This system does not have a solution since alpha_2 < 0')
        elif alpha3 < 0: 
            raise Exception('This system does not have a solution since alpha_3 < 0')
     
        # With the estimated alpha, the value of v2_hat. 
        v2_hat = v1*(1-m2)/(m1*(1-m1))

        err_v2 = abs(v2 - v1*(1-m2)/(m1*(1 - m1)))/v2

        return ((alpha1, alpha2, alpha3, alpha4), v2_hat, err_v2)

class BivariateBeta:

    def __init__(self) -> None:
        pass

    def moments_calculus(self, alpha): 
        
        tilde_alpha = alpha[0]+alpha[1]+alpha[2]+alpha[3]
        
        E_X = (alpha[0]+alpha[1])/tilde_alpha
        E_Y = (alpha[0]+alpha[2])/tilde_alpha
        
        Var_X = (1/(tilde_alpha*(tilde_alpha+1)))*E_X*(alpha[2] + alpha[3])
        Var_Y = (1/(tilde_alpha*(tilde_alpha+1)))*E_Y*(alpha[1] + alpha[3])
        
        den = np.log(alpha[0] + alpha[1]) + np.log(alpha[2]+alpha[3]) + np.log(alpha[0]+alpha[2]) + np.log(alpha[1]+alpha[3])
        den = np.exp(-0.5*den)
        Cor_XY = (alpha[0]*alpha[3] - alpha[1]*alpha[2])*den
        
        return (E_X, E_Y, Var_X, Var_Y, Cor_XY)


    