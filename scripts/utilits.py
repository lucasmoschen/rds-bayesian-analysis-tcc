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

    def minimizer(self, m1, m2, v1, v2, rho, g = None, x0 = (1,1,1,1)): 
        """
        Minimized the loss function as function of alpha. 
        """
        if g is None: 
            g = self.quadratic_error

        return minimize(fun = self.loss_function, 
                        x0 = x0, 
                        args = (m1, m2, v1, v2, rho, g),
                        bounds = [(0, np.inf)]*4,
                        method = 'trust-constr')