import numpy as np
from typing import List, Optional, Sequence, Tuple

from scipy.interpolate import BSpline
from scipy.special import digamma, gammaln


class BABOPDE:
    def __init__(
        self,
        degree: int = 3,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        eps0: float = 1.0,
        eta0: float = 1.0,
        inner_max_iter: int = 100,
        outer_max_iter: int = 20,
        vi_tol: float = 1e-5,
        outer_tol: float = 1e-3,
        prune_tol: float = 0.05,
        mle_max_iter: int = 100,
        mle_tol: float = 1e-6,
        grid_size_for_integral: int = 512,
    ):
        self.degree = degree
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.eps0 = eps0
        self.eta0 = eta0
        self.inner_max_iter = inner_max_iter
        self.outer_max_iter = outer_max_iter
        self.vi_tol = vi_tol
        self.outer_tol = outer_tol
        self.prune_tol = prune_tol
        self.mle_max_iter = mle_max_iter
        self.mle_tol = mle_tol
        self.grid_size_for_integral = grid_size_for_integral
        self.coef_ = None
        self.coef_shape_ = None
        self.active_index_per_dim_ = None
        self.knots_per_dim_ = None
        self.ranges_ = None

    def fit(
        self,
        X: np.ndarray,
        n_basis_init: Sequence[int],
        value_range: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> "BABOPDE":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must have shape (N, D).")

        n_dim = X.shape[1]
        n_basis_init = list(n_basis_init)
        if len(n_basis_init) != n_dim:
            raise ValueError("n_basis_init must contain one value for each dimension.")
        if any(m <= self.degree for m in n_basis_init):
            raise ValueError("Each initial basis count must be greater than degree.")

        if value_range is None:
            value_range = [(X[:, d].min(), X[:, d].max()) for d in range(n_dim)]
        value_range = list(value_range)

        knots_per_dim = [
            self._make_open_uniform_knots(r[0], r[1], m)
            for r, m in zip(value_range, n_basis_init)
        ]
        basis_integrals_per_dim = [
            self._basis_integrals_1d(
                knots_per_dim[d], n_basis_init[d], value_range[d][0], value_range[d][1]
            )
            for d in range(n_dim)
        ]
        basis_values_per_dim = [
            self._basis_matrix_1d(X[:, d], knots_per_dim[d], n_basis_init[d])
            for d in range(n_dim)
        ]

        active_index_per_dim = [np.arange(m) for m in n_basis_init]
        prev_outer_obj = -np.inf

        for _ in range(self.outer_max_iter):
            phi = self._tensor_design_matrix(
                [basis_values_per_dim[d][:, active_index_per_dim[d]] for d in range(n_dim)]
            )
            t_vec = self._tensor_integrals(
                [basis_integrals_per_dim[d][active_index_per_dim[d]] for d in range(n_dim)]
            )
            a_star = self._mle_fixed_point(phi, t_vec)
            dmat = self._build_highdim_difference_operator(
                [len(idx) for idx in active_index_per_dim]
            )
            mu, outer_obj = self._vi_optimize(a_star, dmat)
            coef = self._normalize_coef(np.maximum(mu, 0.0), t_vec)
            active_index_per_dim, changed = self._prune_active_basis_per_dim(
                coef,
                tuple(len(idx) for idx in active_index_per_dim),
                active_index_per_dim,
            )

            if not changed or abs(outer_obj - prev_outer_obj) < self.outer_tol:
                break
            prev_outer_obj = outer_obj

        phi = self._tensor_design_matrix(
            [basis_values_per_dim[d][:, active_index_per_dim[d]] for d in range(n_dim)]
        )
        t_vec = self._tensor_integrals(
            [basis_integrals_per_dim[d][active_index_per_dim[d]] for d in range(n_dim)]
        )
        a_star = self._mle_fixed_point(phi, t_vec)
        dmat = self._build_highdim_difference_operator(
            [len(idx) for idx in active_index_per_dim]
        )
        mu, _ = self._vi_optimize(a_star, dmat)

        self.coef_ = self._normalize_coef(np.maximum(mu, 0.0), t_vec)
        self.coef_shape_ = tuple(len(idx) for idx in active_index_per_dim)
        self.active_index_per_dim_ = active_index_per_dim
        self.knots_per_dim_ = knots_per_dim
        self.ranges_ = value_range
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit before pdf.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.ranges_):
            raise ValueError("X has an incompatible shape.")

        basis_values = []
        for d in range(X.shape[1]):
            n_basis = len(self.knots_per_dim_[d]) - self.degree - 1
            full_basis = self._basis_matrix_1d(
                X[:, d], self.knots_per_dim_[d], n_basis
            )
            basis_values.append(full_basis[:, self.active_index_per_dim_[d]])

        return np.maximum(self._tensor_design_matrix(basis_values) @ self.coef_, 0.0)

    def evaluate_grid(
        self, grid_per_dim: Sequence[np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        mesh = np.meshgrid(*grid_per_dim, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        values = self.pdf(points).reshape(*[len(g) for g in grid_per_dim])
        return tuple(mesh), values

    def _make_open_uniform_knots(self, umin: float, umax: float, n_basis: int) -> np.ndarray:
        n_inner = n_basis - self.degree - 1
        if n_inner > 0:
            inner = np.linspace(umin, umax, n_inner + 2)[1:-1]
            return np.r_[
                np.repeat(umin, self.degree + 1),
                inner,
                np.repeat(umax, self.degree + 1),
            ]
        return np.r_[
            np.repeat(umin, self.degree + 1),
            np.repeat(umax, self.degree + 1),
        ]

    def _basis_matrix_1d(
        self, x: np.ndarray, knots: np.ndarray, n_basis: int
    ) -> np.ndarray:
        values = np.zeros((len(x), n_basis))
        eye = np.eye(n_basis)
        for i in range(n_basis):
            values[:, i] = BSpline(
                knots, eye[i], self.degree, extrapolate=False
            )(x)
        values[np.isnan(values)] = 0.0
        return np.maximum(values, 0.0)

    def _basis_integrals_1d(
        self, knots: np.ndarray, n_basis: int, umin: float, umax: float
    ) -> np.ndarray:
        x = np.linspace(umin, umax, self.grid_size_for_integral)
        return np.trapz(self._basis_matrix_1d(x, knots, n_basis), x, axis=0)

    @staticmethod
    def _tensor_design_matrix(basis_list: List[np.ndarray]) -> np.ndarray:
        phi = basis_list[0]
        for basis in basis_list[1:]:
            phi = np.einsum("ni,nj->nij", phi, basis).reshape(phi.shape[0], -1)
        return phi

    @staticmethod
    def _tensor_integrals(integrals: List[np.ndarray]) -> np.ndarray:
        result = integrals[0]
        for item in integrals[1:]:
            result = np.kron(result, item)
        return result

    def _mle_fixed_point(self, phi: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        n_samples, n_coef = phi.shape
        coef = self._normalize_coef(np.ones(n_coef), t_vec)

        for _ in range(self.mle_max_iter):
            density = phi @ coef + 1e-12
            updated = (coef / (n_samples * np.maximum(t_vec, 1e-12))) * (
                phi / density[:, None]
            ).sum(axis=0)
            updated = self._normalize_coef(np.maximum(updated, 1e-16), t_vec)
            if np.linalg.norm(updated - coef) < self.mle_tol:
                return updated
            coef = updated
        return coef

    @staticmethod
    def _normalize_coef(coef: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        return coef / (np.sum(coef * t_vec) + 1e-12)

    @staticmethod
    def _build_second_diff_1d(n_basis: int) -> np.ndarray:
        if n_basis < 3:
            return np.zeros((0, n_basis))
        dmat = np.zeros((n_basis - 2, n_basis))
        rows = np.arange(n_basis - 2)
        dmat[rows, rows] = 1.0
        dmat[rows, rows + 1] = -2.0
        dmat[rows, rows + 2] = 1.0
        return dmat

    def _build_highdim_difference_operator(self, shape: Sequence[int]) -> np.ndarray:
        operators = []
        for dim in range(len(shape)):
            second_diff = self._build_second_diff_1d(shape[dim])
            if second_diff.shape[0] == 0:
                continue
            operator = np.array([[1.0]])
            for j, size in enumerate(shape):
                operator = np.kron(
                    operator,
                    second_diff if j == dim else np.eye(size),
                )
            operators.append(operator)
        if not operators:
            return np.zeros((0, int(np.prod(shape))))
        return np.vstack(operators)

    def _vi_optimize(self, a_star: np.ndarray, dmat: np.ndarray) -> Tuple[np.ndarray, float]:
        n_coef = len(a_star)
        n_smooth = dmat.shape[0]
        alpha = self.alpha0 + 0.5 * n_coef
        beta = self.beta0 + 1.0
        eps = self.eps0 + 0.5 * max(n_smooth, 1)
        eta = self.eta0 + 1.0
        prev_elbo = -np.inf
        mu = a_star.copy()

        for _ in range(self.inner_max_iter):
            inv_sigma2 = alpha / beta
            inv_rho2 = eps / eta
            precision = inv_sigma2 * np.eye(n_coef)
            if n_smooth > 0:
                precision += inv_rho2 * (dmat.T @ dmat)
            sigma = np.linalg.inv(precision)
            mu = sigma @ (inv_sigma2 * a_star)

            diff2 = (
                a_star @ a_star
                - 2.0 * a_star @ mu
                + np.trace(sigma)
                + mu @ mu
            )
            if n_smooth > 0:
                dtd = dmat.T @ dmat
                smooth2 = np.trace(dtd @ sigma) + mu @ dtd @ mu
            else:
                smooth2 = 0.0

            alpha = self.alpha0 + 0.5 * n_coef
            beta = self.beta0 + 0.5 * diff2
            eps = self.eps0 + 0.5 * max(n_smooth, 1)
            eta = self.eta0 + 0.5 * smooth2

            elbo = self._compute_elbo(
                a_star, mu, sigma, alpha, beta, eps, eta, dmat
            )
            if abs(elbo - prev_elbo) < self.vi_tol:
                return mu, elbo
            prev_elbo = elbo

        return mu, prev_elbo

    def _compute_elbo(
        self,
        a_star: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        alpha: float,
        beta: float,
        eps: float,
        eta: float,
        dmat: np.ndarray,
    ) -> float:
        n_coef = len(a_star)
        n_smooth = dmat.shape[0]
        inv_sigma2 = alpha / beta
        log_sigma2 = np.log(beta) - digamma(alpha)
        inv_rho2 = eps / eta
        log_rho2 = np.log(eta) - digamma(eps)

        diff2 = (
            a_star @ a_star
            - 2.0 * a_star @ mu
            + np.trace(sigma)
            + mu @ mu
        )
        if n_smooth > 0:
            dtd = dmat.T @ dmat
            smooth2 = np.trace(dtd @ sigma) + mu @ dtd @ mu
        else:
            smooth2 = 0.0

        _, logdet_sigma = np.linalg.slogdet(sigma)
        likelihood = -0.5 * n_coef * log_sigma2 - 0.5 * inv_sigma2 * diff2
        smooth_prior = -0.5 * n_smooth * log_rho2 - 0.5 * inv_rho2 * smooth2
        sigma_prior = (
            self.alpha0 * np.log(self.beta0 + 1e-12)
            - gammaln(self.alpha0)
            - (self.alpha0 + 1.0) * log_sigma2
            - self.beta0 * inv_sigma2
        )
        rho_prior = (
            self.eps0 * np.log(self.eta0 + 1e-12)
            - gammaln(self.eps0)
            - (self.eps0 + 1.0) * log_rho2
            - self.eta0 * inv_rho2
        )
        entropy_a = 0.5 * (
            n_coef * (1.0 + np.log(2.0 * np.pi)) + logdet_sigma
        )
        entropy_sigma = -(
            alpha * np.log(beta)
            - gammaln(alpha)
            - (alpha + 1.0) * log_sigma2
            - beta * inv_sigma2
        )
        entropy_rho = -(
            eps * np.log(eta)
            - gammaln(eps)
            - (eps + 1.0) * log_rho2
            - eta * inv_rho2
        )

        return float(
            likelihood
            + smooth_prior
            + sigma_prior
            + rho_prior
            + entropy_a
            + entropy_sigma
            + entropy_rho
        )

    def _prune_active_basis_per_dim(
        self,
        coef: np.ndarray,
        coef_shape: Tuple[int, ...],
        active_index_per_dim: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], bool]:
        coef_tensor = coef.reshape(coef_shape)
        new_active = []
        changed = False

        for dim in range(len(coef_shape)):
            axes = tuple(i for i in range(len(coef_shape)) if i != dim)
            contribution = coef_tensor.sum(axis=axes)
            keep = contribution > self.prune_tol

            if not np.any(keep):
                keep[np.argmax(contribution)] = True

            n_old = len(active_index_per_dim[dim])
            n_min_keep = max(1, int(np.ceil(0.8 * n_old)))
            if keep.sum() < n_min_keep:
                top_idx = np.argsort(contribution)[-n_min_keep:]
                keep = np.zeros_like(keep, dtype=bool)
                keep[top_idx] = True

            if keep.sum() != n_old:
                changed = True
            new_active.append(active_index_per_dim[dim][keep])

        return new_active, changed
