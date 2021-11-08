"""
Base Analysis
"""
from typing import Dict, Any
from abc import abstractmethod, ABCMeta
import numpy as np


class SystemSolver(metaclass=ABCMeta):
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self._param_validation()

    @abstractmethod
    def _param_validation(self):
        pass

    @abstractmethod
    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_u(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    def eval_f_linear(
        self, x: np.ndarray, u: np.ndarray, x0: np.ndarray, u0: np.ndarray
    ) -> np.ndarray:
        # df/du = 1.0 in our case
        return self.eval_f(x0, u0) + self.eval_Jf(x0, u0).dot(x - x0) + 1.0 * (u - u0)

    def eval_Jf_numerical(
        self, x: np.ndarray, u: np.ndarray, dx: float = 1e-7
    ) -> np.ndarray:
        x = x.astype(float)
        f = self.eval_f(x, u)
        J = np.zeros((x.size, x.size))
        for i, _ in enumerate(x):
            delta_x = np.zeros(x.size)
            delta_x[i] = dx
            new_x = x + delta_x
            f_new = self.eval_f(new_x, u)
            delta_f = f_new - f
            J[:, i] = delta_f / dx
        return J

    def forward_euler(self, x_start: np.ndarray, ts: np.ndarray):
        x_start = x_start.astype(float)
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start
        dt = ts[1] - ts[0]
        for i, t in enumerate(ts[:-1]):
            u = self.eval_u(t)
            f = self.eval_f(X[:, i], u)
            X[:, i + 1] = X[:, i] + dt * f
        return X

    def forward_euler_linear(
        self, x_start: np.ndarray, ts: np.ndarray, x0: np.ndarray, u0: np.ndarray
    ):
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start
        dt = ts[1] - ts[0]
        for i, t in enumerate(ts[:-1]):
            u = self.eval_u(t)
            f = self.eval_f_linear(X[:, i], u, x0, u0)
            X[:, i + 1] = X[:, i] + dt * f
        return X

    def trapezoidal(self, x_start: np.ndarray, ts: np.ndarray, **kwargs):
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start
        dt = ts[1] - ts[0]
        ts = np.append(ts, ts[-1] + dt)  # needed for last step

        u = self.eval_u(ts[0])
        u_next = self.eval_u(ts[1])
        for i in range(0, ts.size - 2):
            f = self.eval_f(X[:, i], u)
            x_next_guess = X[:, i] + dt * f  # use Euler as a good initial guess
            p = {"dt": dt, "f": f, "x": X[:, i], "u_next": u_next}
            X[:, i + 1], _ = self.newton(x_next_guess, p, **kwargs)
            u = u_next
            u_next = self.eval_u(ts[i + 2])

        return X

    def eval_f_newton(
        self, x_next: np.ndarray, p: Dict[str, np.ndarray],
    ):
        dt = p["dt"]
        f_curr = p["f"]
        x_curr = p["x"]
        u_next = p["u_next"]

        newton_f = x_curr - x_next + 1 / 2 * dt * (f_curr + self.eval_f(x_next, u_next))
        return newton_f

    def eval_Jf_newton(
        self, x_next: np.ndarray, p: Dict[str, np.ndarray],
    ):
        dt = p["dt"]
        u_next = p["u_next"]

        return -np.eye(x_next.size) + 1 / 2 * dt * self.eval_Jf(x_next, u_next)

    def newton(
        self,
        x0: np.ndarray,
        p: Dict[str, np.ndarray],
        err_f: float = 1e-8,
        err_delta_x: float = 1e-8,
        rel_delta_x=1e-8,
        max_iter=100,
        use_gcr=False,
        finite_difference=False,
        return_iterations=False,
    ):
        X = np.zeros((max_iter + 1, x0.size))

        k = 0
        X[k, :] = x0
        f = self.eval_f_newton(X[k, :], p)
        err_f_k = np.linalg.norm(f, np.inf)

        delta_x = 0
        err_delta_x_k = 0
        rel_delta_x_k = 0

        while k < max_iter and (
            err_f_k > err_f
            or err_delta_x_k > err_delta_x
            or rel_delta_x_k > rel_delta_x
        ):
            if not use_gcr:
                if finite_difference:
                    # Jf = self.eval_Jf_numerical(X[k, :], p)
                    raise NotImplementedError(
                        "Coming soon! Need to generalize eval_Jf_numerical."
                    )
                else:
                    Jf = self.eval_Jf_newton(X[k, :], p)
                delta_x = -np.linalg.solve(Jf, f)
            else:
                delta_x, _ = self.tgcr_matrix_free(-f, X[k, :], p)
            X[k + 1, :] = X[k, :] + delta_x
            k += 1
            f = self.eval_f_newton(X[k, :], p)
            err_f_k = np.linalg.norm(f, np.inf)
            err_delta_x_k = np.linalg.norm(delta_x, np.inf)
            rel_delta_x_k = np.linalg.norm(delta_x, np.inf) / np.max(np.abs(X[k, :]))

        if (
            err_f_k <= err_f
            and err_delta_x_k <= err_delta_x
            and rel_delta_x_k <= rel_delta_x
        ):
            converged = True
        else:
            converged = False

        if return_iterations:
            return X[:k, :], converged
        return X[k - 1, :], converged

    def tgcr_matrix_free(
        self,
        b: np.ndarray,
        xk: np.ndarray,
        params: Dict[str, np.ndarray],
        epsilon: float = 1e-6,
        tol: float = 0.1,
        max_iter: int = 100,
    ):
        """
        Solving Jf delta_x = -f, without Jf matrix
        """
        x = np.zeros_like(b)
        r = b
        f_xk = -b

        r_norms = np.zeros(max_iter + 1)
        r_norms[0] = np.linalg.norm(r, 2)

        p = np.zeros((max_iter + 1, b.size))
        Ap = np.zeros((max_iter + 1, b.size))

        for i in range(max_iter):
            p[i, :] = r
            Ap[i, :] = (
                1.0
                / epsilon
                * (self.eval_f_newton(xk + epsilon * p[i, :], params) - f_xk)
            )

            for j in range(0, i):
                beta = Ap[i, :].T * Ap[j, :]
                p[i, :] = p[i, :] - beta * p[j, :]
                Ap[i, :] = Ap[i, :] - beta * Ap[j, :]

            norm_Ap = np.linalg.norm(Ap[i, :], 2)
            Ap[i, :] = Ap[i, :] / norm_Ap
            p[i, :] = p[i, :] / norm_Ap

            alpha = r.T * Ap[i, :]

            x = x + alpha * p[i, :]
            r = r - alpha * Ap[i, :]

            r_norms[i + 1] = np.linalg.norm(r, 2)

            if r_norms[i + 1] < (tol * r_norms[0]):
                break

        r_norms = r_norms / r_norms[0]
        converged = bool(r_norms[i + 1] < tol)

        return x, converged

    def copy(self):
        """
        Just copies params into another instance of the system class. 
        Not a full copy (so that stored analysis can be reset).
        """
        cls = self.__class__
        return cls(self.params.copy())
