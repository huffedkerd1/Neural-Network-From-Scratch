import numpy as np

class Conv2d:
    def __init__(self, c_in, c_out, k, pad=0, stride=1):
        self.c_in, self.c_out = c_in, c_out
        self.k, self.p, self.s = k, pad, stride

        # -----Learnable Parameters-----
        self.W = np.random.randn(c_out, c_in, k, k) * np.sqrt(2.0 / (c_in * k * k))
        self.b = np.zeros(c_out)

        # -----Gradients-----
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.Xp = None

    def forward(self, X):
        k, s, p = self.k, self.s, self.p
        _, H, W = X.shape

        Xp = np.pad(X, ((0, 0), (p, p), (p, p)))

        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1

        Y = np.zeros((self.c_out, H_out, W_out))

        for f in range(self.c_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = Xp[:, i*s:i*s+k, j*s:j*s+k]
                    Y[f, i, j] = np.sum(patch * self.W[f]) + self.b[f]

        self.Xp = Xp
        return Y

    def backward(self, dY):
        k, s, p = self.k, self.s, self.p
        _, H_out, W_out = dY.shape
        Xp = self.Xp

        self.dW[:] = 0
        self.db[:] = 0

        dXp = np.zeros_like(Xp)

        for f in range(self.c_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = Xp[:, i*s:i*s+k, j*s:j*s+k]
                    g = dY[f, i, j]

                    self.dW[f] += g * patch
                    dXp[:, i*s:i*s+k, j*s:j*s+k] += self.W[f] * g
                    self.db[f] += g

        H_p, W_p = Xp.shape[1], Xp.shape[2]

        if p > 0:
            return dXp[:, p:H_p-p, p:W_p-p]
        else:
            return dXp