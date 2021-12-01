"""
Base Analysis
"""
from typing import Dict, Any
from abc import abstractmethod, ABCMeta
import numpy as np


class SystemSolver(metaclass=ABCMeta):
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params.copy()
        self._param_validation()

    @abstractmethod
    def _param_validation(self):
        self.params["q"] = self.params.get("q", None)
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

    def eval_linear_matrices(self, x0: np.ndarray, u0: np.ndarray):
        if np.isscalar(u0) and u0 == 0:
            u0 = np.zeros_like(x0)

        Jf_0 = self.eval_Jf(x0, u0)
        f_0 = self.eval_f(x0, u0)
        K0 = f_0 - Jf_0.dot(x0) - u0
        K0 = K0.reshape(K0.size, 1)

        Ju_0 = np.eye(u0.size)  # df/du = 1.0 in our case
        A = Jf_0
        B = np.concatenate((K0, Ju_0), axis=1)
        return A, B

    def eval_f_linear(
        self, x: np.ndarray, u: np.ndarray, A: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        u_full = np.append(1, u)
        # print(A.shape)
        # print(x.shape)
        # print(B.shape)
        # print(u_full.shape)
        return A.dot(x) + B.dot(u_full)

    def eval_Jf_numerical(
        self, x: np.ndarray, u: np.ndarray, dx: float = 1e-7
    ) -> np.ndarray:
        return self.eval_numerical_gradient(self.eval_f, x, u, dx=dx,)

    def eval_numerical_gradient(self, eval_f, x: np.ndarray, *args, **kwargs):
        x = x.astype(float)
        dx = kwargs.pop("dx", 1e-7)
        f = eval_f(x, *args, **kwargs)
        J = np.zeros((x.size, x.size))
        for i, _ in enumerate(x):
            delta_x = np.zeros(x.size)
            delta_x[i] = dx
            new_x = x + delta_x
            f_new = eval_f(new_x, *args, **kwargs)
            delta_f = f_new - f
            J[:, i] = delta_f / dx
        return J

    def forward_euler(self, x_start: np.ndarray, ts: np.ndarray):
        x_start = x_start.astype(float)
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start
        for i, t in enumerate(ts[:-1]):
            dt = ts[i + 1] - ts[i]
            u = self.eval_u(t)
            f = self.eval_f(X[:, i], u)
            X[:, i + 1] = X[:, i] + dt * f
        return X

    def forward_euler_linear(
        self, x_start: np.ndarray, ts: np.ndarray, x0: np.ndarray, u0: np.ndarray
    ):
        A, B = self.eval_linear_matrices(x0, u0)
        q = self.params["q"]
        if q is not None:
            evals, V = np.linalg.eig(A)
            V_dag = V.T.conj()
            B_hat = V_dag.dot(B)

            # just take the smallest eigenvalues (i.e. [-q:])
            evals = evals[-q:]
            A_hat = np.diag(evals)
            B_hat = B_hat[
                -q:
            ]  # selecting rows of the B_hat matrix corresponding to smallest eigenvalues
            A, B = A_hat, B_hat
            V_dag_trunc = V_dag[-q:, :]
            x_start = V_dag_trunc.dot(x_start)

        X = 1.0 * np.zeros((x_start.size, ts.size), dtype=complex)
        X[:, 0] = x_start
        for i, t in enumerate(ts[:-1]):
            dt = ts[i + 1] - ts[i]
            u = self.eval_u(t)
            f = self.eval_f_linear(X[:, i], u, A, B)
            X[:, i + 1] = X[:, i] + dt * f

        if q is not None:
            # now we need to convert back to the full basis
            V_trunc = V[:, -q:]
            X = V_trunc.dot(X)

        return np.real(X)

    def trapezoidal_dynamic(
        self, x_start: np.ndarray, ts: np.ndarray, return_last=False, **kwargs
    ):

        # parameters
        factor = kwargs.pop("factor", 2)
        threshold_min = kwargs.pop("threshold_min", 0.5)
        threshold_max = kwargs.pop("threshold_max", 2)

        # initialization
        X = []
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start

        tf = np.max(ts)
        ti = np.min(ts)
        dt_init = ts[1] - ts[0]
        dt = dt_init
        t_curr = ti
        ts_dynamic = []
        i = 0

        u = self.eval_u(ts[0])
        u_next = self.eval_u(ts[1])

        while t_curr <= tf:
            f = self.eval_f(X[:, i], u)

            # UPDATE dt
            if i > 0:  # only update after the first iteration
                denom = np.abs(X[0, i]) ** 2
                if denom > 0:
                    max_perc_change = np.max(np.abs(X[:, i] - X[:, i - 1]) ** 2) / denom
                    slope = max_perc_change / dt
                    if slope < threshold_min:
                        # slow varying
                        dt = dt * factor
                        # print("dt increase: ", dt)
                    elif slope > threshold_max:
                        # fast varying
                        dt = max(dt / factor, dt_init)
                        # print("dt decrease: ", dt)
            ts_dynamic.append(t_curr)
            t_curr += dt

            # solve next time step
            x_next_guess = X[:, i] + dt * f  # use Euler as a good initial guess
            p = {"dt": dt, "f": f, "x": X[:, i], "u_next": u_next}
            X[:, i + 1], _ = self.newton(
                x_next_guess,
                p,
                self.eval_f_trapezoidal,
                self.eval_Jf_trapezoidal,
                **kwargs,
            )
            u = u_next
            if i < ts.size - 2:
                u_next = self.eval_u(ts[i + 2])
            i += 1

        if return_last:
            return X[:, -1]

        ts_dynamic = np.array(ts_dynamic)
        return (X[:, : ts_dynamic.size], ts_dynamic)

    def trapezoidal(
        self,
        x_start: np.ndarray,
        ts: np.ndarray,
        return_last=False,
        dynamic_dt=False,
        **kwargs
    ):
        if dynamic_dt:
            return self.trapezoidal_dynamic(x_start, ts, return_last, **kwargs)

        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start

        u = self.eval_u(ts[0])
        u_next = self.eval_u(ts[1])
        for i in range(0, ts.size - 1):
            f = self.eval_f(X[:, i], u)
            dt = ts[i + 1] - ts[i]
            x_next_guess = X[:, i] + dt * f  # use Euler as a good initial guess
            p = {"dt": dt, "f": f, "x": X[:, i], "u_next": u_next}
            X[:, i + 1], _ = self.newton(
                x_next_guess,
                p,
                self.eval_f_trapezoidal,
                self.eval_Jf_trapezoidal,
                **kwargs,
            )
            u = u_next
            if i < ts.size - 2:
                u_next = self.eval_u(ts[i + 2])

        if return_last:
            return X[:, -1]

        return X

    def eval_f_shooting_newton(self, x: np.ndarray, p: Dict[str, np.ndarray]):
        ts = p["ts"]
        return self.trapezoidal(x, ts, return_last=True) - x

    def eval_Jf_shooting_newton_numerical(
        self, x: np.ndarray, p: Dict[str, np.ndarray]
    ):
        ts = p["ts"]
        Jf = self.eval_numerical_gradient(
            self.trapezoidal, x, ts, return_last=True
        ) - np.eye(x.size)
        return Jf

    def eval_solve_shooting_newton(self, x0: np.ndarray, ts: np.ndarray, **kwargs):
        p = {"ts": ts}
        return self.newton(
            x0,
            p,
            self.eval_f_shooting_newton,
            self.eval_Jf_shooting_newton_numerical,
            **kwargs,
        )[0]

    def eval_f_trapezoidal(
        self, x_next: np.ndarray, p: Dict[str, np.ndarray],
    ):
        dt = p["dt"]
        f_curr = p["f"]
        x_curr = p["x"]
        u_next = p["u_next"]

        newton_f = x_curr - x_next + 1 / 2 * dt * (f_curr + self.eval_f(x_next, u_next))
        return newton_f

    def eval_Jf_trapezoidal(
        self, x_next: np.ndarray, p: Dict[str, np.ndarray],
    ):
        dt = p["dt"]
        u_next = p["u_next"]

        return -np.eye(x_next.size) + 1 / 2 * dt * self.eval_Jf(x_next, u_next)

    def newton(
        self,
        x0: np.ndarray,
        p: Dict[str, np.ndarray],
        eval_f,
        eval_Jf,
        err_f: float = 1e-8,
        err_delta_x: float = 1e-8,
        rel_delta_x=1e-8,
        max_iter=100,
        use_gcr=False,
        finite_difference=False,
        return_iterations=False,
        **kwargs
    ):
        """
        The newton method is used to find the zeros of a nonlinear function.
        Here, we solve for the zero of a particular function used for the Trapezoidal method.

        Arguments:
            x0 (np.ndarray): initial guess
            p (Dict[str, np.ndarray]):
                key: description of array
                value: array used in evaluating self.eval_f_newton
            eval_f (function):
                function used to calculate f
            eval_Jf (function):
                function used to calculate gradient of f
            err_f (float): ||f||_{inf} <= err_f condition for convergence
            err_delta_x (float): ||delta_x||_{inf} <= err_f condition for convergence
            rel_delta_x (float): ||delta_x||_{inf}/max(|X_k|) <= err_f condition for convergence
            max_iter (int): maximum iterations of newton that can be run
            use_gcr (bool): whether to use GCR instead of calculating Jf to find delta_x from Jf delta_x = -f
            finite_difference (bool): whetherr to use finite difference to calculate Jf
            return_iterations (bool): whether to return all intermediate solutions or just the final solution

        Returns:
            (all) X[:k,:] or (final) X[k-1,:]:
                all intermediate solutions or just the final solution
            converged (bool): whether Newton converged or not
        """
        X = np.zeros((max_iter + 1, x0.size))

        k = 0
        X[k, :] = x0
        f = eval_f(X[k, :], p)
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
                    Jf = eval_Jf(X[k, :], p)
                delta_x = -np.linalg.solve(Jf, f)
            else:
                delta_x, _ = self.tgcr_matrix_free(-f, X[k, :], p, eval_f, **kwargs)
            X[k + 1, :] = X[k, :] + delta_x
            k += 1
            f = eval_f(X[k, :], p)
            err_f_k = np.linalg.norm(f, np.inf)
            err_delta_x_k = np.linalg.norm(delta_x, np.inf)
            rel_delta_x_k = np.linalg.norm(delta_x, np.inf) / np.max(np.abs(X[k, :]))

        converged = bool(
            err_f_k <= err_f
            and err_delta_x_k <= err_delta_x
            and rel_delta_x_k <= rel_delta_x
        )

        if return_iterations:
            return X[:k, :], converged
        return X[k - 1, :], converged

    def tgcr_matrix_free(
        self,
        b: np.ndarray,
        xk: np.ndarray,
        params: Dict[str, np.ndarray],
        eval_f,
        epsilon: float = 1e-6,
        tol: float = 0.1,
        max_iter: int = 100,
    ):
        """
        the Generalized Conjugate Residual (tGCR) method

        Solving Jf delta_x = -f, without Jf matrix using the matrix-free or implicit method.

        Arguments:
            b (np.ndarray): right hand side of Ax = b
            xk (np.ndarray): X[k,:] from Newton's method implementation in self.newton
            params (Dict[str,np.ndarray]):
                key: description of array
                value: array used in evaluating self.eval_f_newton
            eval_f (function):
                function used to calculate f
            epsilon (float): step size
            tol (float): convergence tolerance
            max_iter (int): maximum number of iterations

        Returns:
            x (np.ndarray): soluition to Ax = b
            converged (bool): whether GCR converged or not
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
            Ap[i, :] = 1.0 / epsilon * (eval_f(xk + epsilon * p[i, :], params) - f_xk)

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
