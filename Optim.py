import numpy as np

class optim:
    def __init__(self, fun, gfun, x0, method='bfgs'):
        self.fun = fun
        self.gfun = gfun
        self.method = method
        if self.method == 'bfgs':
            self.result = self.bfgs(self.fun, self.gfun, x0)
        elif self.method == 'dfp':
            self.result = self.dfp(self.fun, self.gfun, x0)

    def bfgs(self, fun, gfun, x0):
        result = []
        # 确定参数
        epsilon = 1e-6
        x_size = x0.shape[0]
        Hk = np.eye(x_size)
        rho = 0.2
        sigma = 0.6
        # 进入循环
        m = 0
        while m < 10000:
            # 计算搜索方向
            gx = gfun(x0)
            pk = - np.linalg.solve(Hk, gx)  # 是
            # 计算步长
            i = 0
            great_i = 0
            while i < 20:
                oldf = fun(x0)
                newf = fun(x0 + rho ** i * pk)
                if newf < oldf + sigma * rho ** i * np.dot(gx.T, pk):
                    great_i = i
                    break
                i += 1

            # 更新
            x = x0 + rho ** great_i * pk
            sk = x - x0
            yk = gfun(x) - gx

            denom = np.dot(sk.T, yk)
            if abs(denom) < 1e-10:
                denom = 1e-10

            Hk = Hk + (1 + np.linalg.multi_dot([yk.T, Hk, yk]) / denom) \
                 * np.dot(sk, sk.T) / denom \
                 - (np.linalg.multi_dot([sk, yk.T, Hk]) + np.linalg.multi_dot([Hk, yk, sk.T])) / denom

            x0 = x
            m += 1

            # 检查收敛
            if np.linalg.norm(gfun(x)) < epsilon:
                break
            if m % 1000 == 0:
                result.append(fun(x))

        return result

    def dfp(self, fun, gfun, x0):
        result = []
        # 确定参数 - 步长要搜索 rho，c 初始梯度，初始hessian矩阵
        epsilon = 1e-6
        x_size = x0.shape[0]
        Hk = np.eye(x_size)

        rho = 0.55
        sigma = 0.4
        # 进入循环
        m = 0
        while m < 100:
            # 确定搜索方向
            gx = gfun(x0)
            pk = np.dot(-Hk, gx)  # 2*2 · 2*1 -> 2*1
            # 线搜索步长
            i = 0
            great_i = 0
            while i < 20:
                oldf = fun(x0)
                newf = fun(x0 + rho ** i * pk)
                if (newf < oldf + sigma * (rho ** i) * (np.dot(gx.T, pk))):
                    great_i = i
                    break
                i += 1

            # 最优步长后更新参数
            x = x0 + rho ** great_i * pk
            sk = x - x0
            yk = gfun(x) - gx
            # 更新Hessian矩阵 -- 这样可以表示
            Hk = Hk + np.dot(sk, sk.T) / np.dot(sk.T, yk) - np.linalg.multi_dot(
                [Hk, yk, yk.T, Hk]) / np.linalg.multi_dot([yk.T, Hk, yk])
            x0 = x

            # 循环更新
            m += 1

            if np.linalg.norm(gfun(x)) < epsilon:
                break
            result.append(fun(x))

        return result

    def get_result(self):
        return self.result


def fun(x):
    return 100 * (x[0,0] ** 2 - x[1,0]) ** 2 + (x[0,0] - 1) ** 2

def gfun(x):
    grad = np.zeros_like(x)
    grad[0,0] = 400 * (x[0,0] ** 2 - x[1,0]) * x[0,0] + 2 * (x[0,0] - 1)
    grad[1,0] = -22 * (x[0,0] ** 2 - x[1,0])
    return grad

optimizer = optim(fun,gfun,x0,'dfp')
optimizer.get_result()