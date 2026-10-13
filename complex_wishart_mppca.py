"""
Experimental Complex-Wishart MP-PCA for complex-valued MRI data.

For a mean-centered complex Gaussian noise patch Xc with N spatial samples
and p complex channels, W = Xc^H Xc follows a central complex Wishart law
with degrees of freedom nu=N-1. This implementation uses the finite-size
largest-eigenvalue distribution of CW_p(nu, I) as a noise threshold.

Assumptions:
- locally white circular complex Gaussian noise
- channel covariance proportional to identity
- complex variance convention sigma_c^2 = E[|n|^2]

This is an experimental research implementation, not an official DIPY method.
"""

from functools import lru_cache
import copy
import numpy as np
import mpmath as mp


def _patch_radius_array(arr, patch_radius):
    pr = copy.deepcopy(patch_radius)
    if isinstance(pr, int):
        pr = np.ones(3, dtype=int) * pr
    pr = np.asarray(pr, dtype=int)
    if pr.size != 3:
        raise ValueError("patch_radius must be an int or length-3 array.")
    pr[np.asarray(arr.shape[:3]) == 1] = 0
    return pr


def _complex_wishart_lambda_max_cdf(x, nu, p):
    """
    Exact finite-dimensional CDF of lambda_max for
    W ~ CW_p(nu, I), nu >= p.

    F(x) =
      det[ gamma_lower(nu-p+i+j+1, x) ] /
      det[ Gamma(nu-p+i+j+1) ].
    """
    if x <= 0:
        return mp.mpf("0")
    if nu < p:
        raise ValueError("Complex Wishart CDF requires nu >= p.")

    A = mp.matrix(p)
    B = mp.matrix(p)

    for i in range(p):
        for j in range(p):
            a = nu - p + i + j + 1
            A[i, j] = mp.gammainc(a, 0, x)
            B[i, j] = mp.gamma(a)

    val = mp.det(A) / mp.det(B)
    return min(mp.mpf("1"), max(mp.mpf("0"), val))


@lru_cache(maxsize=None)
def complex_wishart_lambda_max_quantile(nu, p, alpha=0.99):
    """
    q satisfying P(lambda_max(W) <= q)=alpha,
    W ~ CW_p(nu, I).
    """
    nu = int(nu)
    p = int(p)
    alpha = float(alpha)

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    if nu < p:
        raise ValueError("nu must be >= p.")

    mp.mp.dps = 50

    edge = (np.sqrt(nu) + np.sqrt(p)) ** 2
    lo = mp.mpf("0")
    hi = mp.mpf(str(max(1.0, 1.5 * edge)))
    target = mp.mpf(str(alpha))

    while _complex_wishart_lambda_max_cdf(hi, nu, p) < target:
        hi *= 2

    for _ in range(100):
        mid = (lo + hi) / 2
        if _complex_wishart_lambda_max_cdf(mid, nu, p) < target:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2)


def complex_wishart_classifier(eigvals, n_samples, alpha=0.99):
    """
    Classify ascending covariance eigenvalues as noise/signal.

    eigvals are eigenvalues of C = Xc^H Xc / N.
    Mean subtraction gives nu=N-1 degrees of freedom.

    Returns
    -------
    sigma2_complex : float
        Estimate of E[|n|^2].
    n_noise : int
        Number of smallest eigenvalues classified as noise.
    threshold : float
        Final threshold in covariance-eigenvalue units.
    """
    d = np.asarray(eigvals, dtype=np.float64)
    d = np.maximum(d, 0.0)

    if d.ndim != 1:
        raise ValueError("eigvals must be 1-D.")

    N = int(n_samples)
    nu = N - 1
    p_total = d.size

    if nu < 1:
        raise ValueError("Need at least 2 spatial samples.")

    max_k = min(p_total, nu)

    for k in range(max_k, 0, -1):
        residual = d[:k]

        # Since d comes from Xc^H Xc / N but E[Xc^H Xc]=nu*sigma^2 I:
        sigma2_complex = (N / nu) * float(np.mean(residual))

        q = complex_wishart_lambda_max_quantile(
            nu=nu,
            p=k,
            alpha=alpha,
        )

        threshold = sigma2_complex * q / N

        if residual[-1] <= threshold:
            return sigma2_complex, k, threshold

    sigma2_complex = (N / nu) * float(d[0])
    q = complex_wishart_lambda_max_quantile(nu=nu, p=1, alpha=alpha)
    threshold = sigma2_complex * q / N
    return sigma2_complex, 1, threshold


def complex_wishart_mppca(
    arr,
    *,
    mask=None,
    patch_radius=2,
    alpha=0.99,
    return_sigma=False,
    return_ncomps=False,
    out_dtype=None,
):
    """
    Experimental Direct Complex MP-PCA with finite-size
    Complex-Wishart largest-eigenvalue thresholding.
    """
    arr = np.asarray(arr)

    if arr.ndim != 4:
        raise ValueError("arr must be 4D: (X,Y,Z,C).")
    if not np.iscomplexobj(arr):
        raise ValueError("Input must be complex-valued.")

    if out_dtype is None:
        out_dtype = arr.dtype

    if mask is None:
        mask = np.ones(arr.shape[:3], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:3]:
            raise ValueError("mask shape must equal arr.shape[:3].")

    pr = _patch_radius_array(arr, patch_radius)
    patch_size = 2 * pr + 1
    N = int(np.prod(patch_size))
    p = arr.shape[-1]

    if N - 1 < p:
        raise ValueError(
            f"Need N-1 >= channels; got N={N}, channels={p}."
        )

    calc_dtype = np.complex128 if arr.dtype == np.complex128 else np.complex64

    theta = np.zeros(arr.shape, dtype=np.float64)
    thetax = np.zeros(arr.shape, dtype=calc_dtype)

    sigma_acc = np.zeros(arr.shape[:3], dtype=np.float64)
    sigma_w = np.zeros(arr.shape[:3], dtype=np.float64)

    nsignal_acc = np.zeros(arr.shape[:3], dtype=np.float64)
    nsignal_w = np.zeros(arr.shape[:3], dtype=np.float64)

    for k0 in range(pr[2], arr.shape[2] - pr[2]):
        for j0 in range(pr[1], arr.shape[1] - pr[1]):
            for i0 in range(pr[0], arr.shape[0] - pr[0]):

                if not mask[i0, j0, k0]:
                    continue

                ix1, ix2 = i0 - pr[0], i0 + pr[0] + 1
                jy1, jy2 = j0 - pr[1], j0 + pr[1] + 1
                kz1, kz2 = k0 - pr[2], k0 + pr[2] + 1

                patch = arr[ix1:ix2, jy1:jy2, kz1:kz2]
                X = patch.reshape(N, p).astype(calc_dtype, copy=False)

                mean = np.mean(X, axis=0)
                Xc = X - mean

                # Hermitian covariance
                C = (Xc.conj().T @ Xc) / N
                d, W = np.linalg.eigh(C)
                d = np.maximum(d.real, 0.0)

                sigma2_c, n_noise, threshold = complex_wishart_classifier(
                    d,
                    n_samples=N,
                    alpha=alpha,
                )

                Wd = W.copy()
                Wd[:, :n_noise] = 0

                Xest = Xc @ Wd @ Wd.conj().T + mean
                Xest = Xest.reshape(*patch_size, p)

                n_signal = p - n_noise
                this_theta = 1.0 / (1.0 + n_signal)

                theta[ix1:ix2, jy1:jy2, kz1:kz2] += this_theta
                thetax[ix1:ix2, jy1:jy2, kz1:kz2] += Xest * this_theta

                sigma_acc[ix1:ix2, jy1:jy2, kz1:kz2] += (
                    np.sqrt(sigma2_c) * this_theta
                )
                sigma_w[ix1:ix2, jy1:jy2, kz1:kz2] += this_theta

                nsignal_acc[ix1:ix2, jy1:jy2, kz1:kz2] += (
                    n_signal * this_theta
                )
                nsignal_w[ix1:ix2, jy1:jy2, kz1:kz2] += this_theta

    den = np.zeros_like(thetax)
    valid = theta[..., 0] > 0
    den[valid] = thetax[valid] / theta[valid]
    den[~mask] = 0

    outputs = [den.astype(out_dtype)]

    if return_sigma:
        sigma_map = np.zeros(arr.shape[:3], dtype=np.float32)
        ok = sigma_w > 0
        sigma_map[ok] = (sigma_acc[ok] / sigma_w[ok]).astype(np.float32)
        sigma_map[~mask] = 0
        outputs.append(sigma_map)

    if return_ncomps:
        nsignal_map = np.zeros(arr.shape[:3], dtype=np.float32)
        ok = nsignal_w > 0
        nsignal_map[ok] = (nsignal_acc[ok] / nsignal_w[ok]).astype(np.float32)
        nsignal_map[~mask] = 0
        outputs.append(nsignal_map)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)
