import numpy as np

#パラメータの編集、調整の唯一許されたクラスを作成
class Parameter:
    def __init__(self,explain,depend,numExplain,eta_b,eta_w):
        #調整すべきパラメータb:切片、w:d次元分の傾きを作成
        self.w = np.zeros(numExplain)
        self.b = 0
        #説明変数X(d次元列ベクトルn個分)
        self.X = explain
        #目的変数y(n個分の0か1のラベル)
        self.y = depend
        #bとwの学習率
        self.eta_b = eta_b
        self.eta_w = eta_w

    
    def grad(self):
        w_abs = np.sqrt(self.w @ self.w)
        if w_abs == 0.0:
            w_abs = 1
        
        exp = np.exp(-(self.w @ self.X + w_abs * self.b))
        u = self.y - 1/(1 + exp)
        grad_b = w_abs * np.sum(u)
        grad_w = u @ (self.X.T + (self.b / w_abs) * self.w)
        grad_w = np.reshape(grad_w, np.shape(self.w))
        return grad_b, grad_w
    
    def shift(self):
        grad_b, grad_w = self.grad()
        print(grad_b, grad_w)
        self.b += self.eta_b * grad_b
        self.w += self.eta_w * grad_w

    def get_b(self):
        return self.b
    
    def get_w(self):
        return self.w[0]


if __name__ == "__main__":
    
    X = np.array([[1.0,1.0,2.0,2.0,-1.0,-1.0,-2.0,-2.0],[1.0,2.0,1.0,2.0,-1.0,-2.0,-1.0,-2.0]])
    y = np.array([1,1,1,1,0,0,0,0])
    
    p = Parameter(X,y,2,0.1,0.01)
    print(p.grad())