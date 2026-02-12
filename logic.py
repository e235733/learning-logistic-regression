import numpy as np


class Parameter:
    def __init__(self,explain,depend,numExplain,eta_b,eta_w):
        self.w = np.zeros(numExplain)
        self.b = 0.0
        self.X = explain
        self.y = depend
        self.eta_b = eta_b
        self.eta_w = eta_w
    
    def grad(self):
        exp = np.exp(-(self.w @ self.X + self.b))
        u = self.y - 1/(1 + exp)
        grad_b = np.sum(u)
        grad_w = u @ self.X.T
        return grad_b, grad_w
    
    def sift(self):
        grad_b, grad_w = self.grad()
        self.b -= self.eta_b * grad_b
        self.w -= self.eta_w * grad_w

    def get_b(self):
        return self.b
    
    def get_w(self):
        return self.w
    


if __name__ == "__main__":
    
    X = np.array([[1.0,1.0,2.0,2.0,-1.0,-1.0,-2.0,-2.0],[1.0,2.0,1.0,2.0,-1.0,-2.0,-1.0,-2.0]])
    y = np.array([1.0,1.0,1.0,1.0,0,0,0,0])
    
    p = Parameter(X,y,2,0.1,0.01)
    print(p.grad())