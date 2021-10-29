# -*- coding: utf-8 -*-
import numpy as np
from tensorly.decomposition import parafac
import jax
import jax.numpy as jnp
import gc


cdist = lambda XA, XB: np.sum((XA[:, None] - XB[None, :])**2, axis=2)


class ManifoldModeling:
    def __init__(self, X, latent_dim, resolution, sigma_max, sigma_min, tau, model_name, model=None, gamma=None, init='parafac'):

        # 入力データXについて
        if X.ndim == 2:
            self.X = X.reshape((X.shape[0], X.shape[1], 1))
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元

        elif X.ndim == 3:
            self.X = X
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元
        else:
            raise ValueError("invalid X: {}\nX must be 2d or 3d ndarray".format(X))

        # 最大近傍半径(SIGMAX)の設定
        if type(sigma_max) is float:
            self.SIGMA1_MAX = sigma_max
            self.SIGMA2_MAX = sigma_max
        elif isinstance(sigma_max, (list, tuple)):
            self.SIGMA1_MAX = sigma_max[0]
            self.SIGMA2_MAX = sigma_max[1]
        else:
            raise ValueError("invalid sigma_max: {}".format(sigma_max))

        # 最小近傍半径(sigma_min)の設定
        if type(sigma_min) is float:
            self.SIGMA1_MIN = sigma_min
            self.SIGMA2_MIN = sigma_min
        elif isinstance(sigma_min, (list, tuple)):
            self.SIGMA1_MIN = sigma_min[0]
            self.SIGMA2_MIN = sigma_min[1]
        else:
            raise ValueError("invalid sigma_min: {}".format(sigma_min))

        # 時定数(tau)の設定
        if type(tau) is int:
            self.tau1 = tau
            self.tau2 = tau
        elif isinstance(tau, (list, tuple)):
            self.tau1 = tau[0]
            self.tau2 = tau[1]
        else:
            raise ValueError("invalid tau: {}".format(tau))

        # 潜在空間の設定
        self.resoluton = resolution
        self.latent_dim1 = latent_dim
        self.latent_dim2 = latent_dim
        zeta = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
        self.Zeta1 = np.dstack(zeta).reshape(resolution**2, latent_dim)
        self.Zeta2 = np.dstack(zeta).reshape(resolution**2, latent_dim)

        self.K1 = self.Zeta1.shape[0]
        self.K2 = self.Zeta2.shape[0]

        # 勝者ノードの初期化
        model_z = parafac(self.X.reshape(self.N1, self.N2), rank=2)
        self.Z1, self.Z2 = model_z.factors

        self.history = {}

    def fit(self, nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.observed_dim))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim2))
        self.history['sigma1'] = np.zeros(nb_epoch)
        self.history['sigma2'] = np.zeros(nb_epoch)
        self.history['sigma'] = np.zeros(nb_epoch)

        self.X = jnp.array(self.X)
        self.Y = None
        self.Z1 = jnp.array(self.Z1)
        self.Z2 = jnp.array(self.Z2)
        self.Zeta1 = jnp.array(self.Zeta1)
        self.Zeta2 = jnp.array(self.Zeta2)

        for epoch in range(nb_epoch):
            # 学習量の決定
            sigma1 = max(self.SIGMA1_MIN, self.SIGMA1_MIN + (self.SIGMA1_MAX - self.SIGMA1_MIN) * (1 - (epoch / self.tau1)))
            sigma2 = max(self.SIGMA2_MIN, self.SIGMA2_MIN + (self.SIGMA2_MAX - self.SIGMA2_MIN) * (1 - (epoch / self.tau2)))
            self.Y, self.Z1, self.Z2 = fit_once(self.X, self.Y, self.Z1, self.Z2, self.Zeta1, self.Zeta2, sigma1, sigma2)

            self.history['y'][epoch, :, :] = np.array(self.Y)
            self.history['z1'][epoch, :] = np.array(self.Z1)
            self.history['z2'][epoch, :] = np.array(self.Z2)
            self.history['sigma1'][epoch] = sigma1
            self.history['sigma2'][epoch] = sigma2
            self.history['sigma'][epoch] = sigma2
        gc.collect()


# @partial(jax.jit, static_argnums=(6, 7))
@jax.jit
def fit_once(X, Y, Z1, Z2, Zeta1, Zeta2, sigma1, sigma2):
    distance1 = cdist(Zeta1, Z1)  # 距離行列をつくるDはN*K行列
    H1 = jnp.exp(-distance1 / (2 * sigma1**2))  # かっこに気を付ける
    G1 = jnp.sum(H1, axis=1)  # Gは行ごとの和をとったベクトル
    R1 = (H1.T / G1).T  # 行列の計算なので.Tで転置を行う

    distance2 = cdist(Zeta2, Z2)  # 距離行列をつくるDはN*K行列
    H2 = jnp.exp(-distance2 / (2 * sigma2**2))  # かっこに気を付ける
    G2 = jnp.sum(H2, axis=1)  # Gは行ごとの和をとったベクトル
    R2 = (H2.T / G2).T  # 行列の計算なので.Tで転置を行う

    #２次モデルの決定
    Y = jnp.einsum('ki,lj,ijd->kld', R1, R2, X)
    # １次モデル，２次モデルの決定
    U = jnp.einsum('lj,ijd->ild', R2, X)
    V = jnp.einsum('ki,ijd->kjd', R1, X)

    # 勝者決定
    k_star1 = jnp.argmin(jnp.sum(jnp.square(U[:, None, :, :] - Y[None, :, :, :]), axis=(2, 3)), axis=1)
    k_star2 = jnp.argmin(jnp.sum(jnp.square(V[:, :, None, :] - Y[:, None, :, :]), axis=(0, 3)), axis=1)

    Z1 = Zeta1[k_star1, :]  # k_starのZの座標N*L(L=2
    Z2 = Zeta2[k_star2, :]  # k_starのZの座標N*L(L=2

    return Y, Z1, Z2


if __name__ == '__main__':
    from time import time


    nb_epoch = 50
    sigma_max = 2.2
    sigma_min = 0.2
    tau = 50
    latent_dim = 2
    seed = 1
    resolution=10
    model_name="TSOM"

    X = np.array(np.arange(100*600).reshape(100, 600), dtype=np.float64)

    start = time()
    mm = ManifoldModeling(
            X,
            latent_dim=latent_dim,
            resolution=resolution,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            model_name=model_name,
            tau=tau,
            init='parafac'
        )
    mm.fit(nb_epoch=nb_epoch)
    print(f"duration: {time() - start}")
