
"""
Hybrid Multi-Echo Complex MP-PCA
--------------------------------
This implementation reproduces the heuristic hybrid method described in
the uploaded project report.

Core components:
- Direct complex-valued local PCA
- Theoretical MP threshold
- Empirical robust-noise threshold
- Echo-wise robust noise estimation with MAD outlier rejection
- Phase coherence
- Temporal coherence
- Echo consistency factor
- Rician-style correction factor
- Safety margin
- Confidence-weighted patch aggregation

IMPORTANT:
This is an experimental heuristic extension, not standard MP-PCA and not an
official DIPY implementation.
"""

import copy
import numpy as np

try:
    from scipy.stats import median_abs_deviation
except Exception:
    def median_abs_deviation(x):
        x = np.asarray(x)
        med = np.median(x)
        return np.median(np.abs(x - med))


EPS = 1e-10


def _patch_radius_array(arr, patch_radius):
    pr = copy.deepcopy(patch_radius)
    if isinstance(pr, int):
        pr = np.ones(3, dtype=int) * pr
    pr = np.asarray(pr, dtype=int)
    if pr.size != 3:
        raise ValueError("patch_radius must be int or length-3 sequence.")
    pr[np.asarray(arr.shape[:3]) == 1] = 0
    return pr


def _compute_phase_coherence(data):
    """
    Report-style phase coherence:
      local_variation = sqrt(dx^2 + dy^2)
      coherence = 1 / (1 + local_variation)
    averaged over echoes.

    For 3D data, x/y gradients are used exactly as in the report.
    """
    phase = np.angle(data)
    X, Y, Z, Ne = phase.shape

    coherence_map = np.zeros((X, Y, Z), dtype=np.float32)

    for echo in range(Ne):
        phase_echo = phase[..., echo]

        dx = np.gradient(phase_echo, axis=0)
        dy = np.gradient(phase_echo, axis=1)

        dx = np.angle(np.exp(1j * dx))
        dy = np.angle(np.exp(1j * dy))

        local_variation = np.sqrt(dx**2 + dy**2)
        coherence_map += (1.0 / (1.0 + local_variation)).astype(np.float32)

    return coherence_map / max(Ne, 1)


def _compute_temporal_coherence(data):
    """
    Report-style temporal coherence:
      diff = |mag2-mag1| / (mag1+mag2+eps)
      coherence = 1 / (1 + diff)
    averaged across adjacent echoes.
    """
    magnitude = np.abs(data)
    X, Y, Z, Ne = magnitude.shape

    coherence_map = np.zeros((X, Y, Z), dtype=np.float32)

    if Ne < 2:
        return np.ones((X, Y, Z), dtype=np.float32)

    for i in range(Ne - 1):
        mag1 = magnitude[..., i]
        mag2 = magnitude[..., i + 1]

        diff = np.abs(mag2 - mag1) / (mag1 + mag2 + EPS)
        coherence_map += (1.0 / (1.0 + diff)).astype(np.float32)

    return coherence_map / (Ne - 1)


def _estimate_echo_noise(X_centered):
    """
    Heuristic echo-wise noise estimate.

    The report indicates echo-wise independent noise estimation, but does not
    fully define _estimate_echo_noise(). For a single centered complex echo
    vector, RMS complex amplitude is used as the local noise proxy.

    Returns a scalar standard-deviation-like quantity.
    """
    X_centered = np.asarray(X_centered)
    return float(np.sqrt(np.mean(np.abs(X_centered) ** 2)))


def _robust_echo_noise(X_centered, z_thresh=2.5):
    """
    Echo-wise noise -> MAD outlier rejection -> mean of valid echoes.
    """
    Ne = X_centered.shape[1]
    echo_noise_estimates = []

    for echo in range(Ne):
        X_echo = X_centered[:, echo:echo + 1]
        noise_est = _estimate_echo_noise(X_echo)
        echo_noise_estimates.append(noise_est)

    echo_noise_array = np.asarray(echo_noise_estimates, dtype=np.float64)

    median_noise = np.median(echo_noise_array)
    mad_noise = median_abs_deviation(echo_noise_array)

    z_scores = np.abs(echo_noise_array - median_noise) / (mad_noise + EPS)
    valid_echoes = z_scores < z_thresh

    if not np.any(valid_echoes):
        valid_echoes[:] = True

    robust_noise = float(np.mean(echo_noise_array[valid_echoes]))

    return robust_noise, echo_noise_array, valid_echoes


def _theoretical_noise_from_eigs(eigenvalues):
    """
    MP-like theoretical noise estimate from lower eigenvalues.

    The report states sigma_theoretical is estimated from the eigenvalue
    distribution but does not provide an exact estimator in the hybrid section.
    We use the lower half median as a robust theoretical estimate.
    """
    d = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    k = max(1, len(d) // 2)
    return float(np.median(d[:k]))


def hybrid_multi_echo_mppca(
    arr,
    *,
    mask=None,
    patch_radius=2,
    empirical_factor=2.0,
    temporal_weight=0.3,
    phase_weight=0.2,
    consistency_weight=0.2,
    safety_margin=1.1,
    echo_outlier_z=2.5,
    use_rician_correction=True,
    return_maps=False,
    out_dtype=None,
):
    """
    Hybrid Multi-Echo Complex MP-PCA.

    Parameters
    ----------
    arr : complex ndarray, shape (X,Y,Z,Ne)
        Multi-echo complex MRI data.
    mask : bool ndarray, shape (X,Y,Z), optional
    patch_radius : int or length-3 sequence
    empirical_factor : float
        Report value: 2.0
    temporal_weight : float
        Report value: 0.3
    phase_weight : float
        Report value: 0.2
    consistency_weight : float
        Report value: 0.2
    safety_margin : float
        Report value: 1.1
    echo_outlier_z : float
        MAD-based echo outlier threshold. Report value: 2.5
    use_rician_correction : bool
    return_maps : bool
        If True, also return diagnostic maps.
    out_dtype : dtype, optional

    Returns
    -------
    denoised
    or
    denoised, maps
    """

    arr = np.asarray(arr)

    if arr.ndim != 4:
        raise ValueError("arr must be 4D: (X,Y,Z,Ne).")
    if not np.iscomplexobj(arr):
        raise ValueError("arr must be complex-valued.")

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
    n_voxels = int(np.prod(patch_size))
    Ne = arr.shape[-1]

    if n_voxels <= Ne:
        raise ValueError(
            f"Patch contains {n_voxels} voxels but Ne={Ne}. "
            "Use a larger patch."
        )

    phase_coherence = _compute_phase_coherence(arr)
    temporal_coherence = _compute_temporal_coherence(arr)

    theta = np.zeros(arr.shape, dtype=np.float64)
    thetax = np.zeros(arr.shape, dtype=arr.dtype)

    threshold_map = np.zeros(arr.shape[:3], dtype=np.float32)
    confidence_map = np.zeros(arr.shape[:3], dtype=np.float32)
    nsignal_map = np.zeros(arr.shape[:3], dtype=np.float32)
    robust_noise_map = np.zeros(arr.shape[:3], dtype=np.float32)

    for k0 in range(pr[2], arr.shape[2] - pr[2]):
        for j0 in range(pr[1], arr.shape[1] - pr[1]):
            for i0 in range(pr[0], arr.shape[0] - pr[0]):

                if not mask[i0, j0, k0]:
                    continue

                ix1, ix2 = i0 - pr[0], i0 + pr[0] + 1
                jy1, jy2 = j0 - pr[1], j0 + pr[1] + 1
                kz1, kz2 = k0 - pr[2], k0 + pr[2] + 1

                patch = arr[ix1:ix2, jy1:jy2, kz1:kz2]
                X = patch.reshape(n_voxels, Ne)

                mean = np.mean(X, axis=0)
                X_centered = X - mean

                # Complex SVD
                U, S, Vh = np.linalg.svd(
                    X_centered,
                    full_matrices=False
                )

                eigenvalues = (S ** 2) / n_voxels

                # ---- 1) theoretical threshold
                sigma_theoretical2 = _theoretical_noise_from_eigs(eigenvalues)
                gamma = Ne / n_voxels

                lambda_theoretical = (
                    sigma_theoretical2
                    * (1.0 + np.sqrt(gamma)) ** 2
                )

                # ---- 2) echo-wise robust noise + empirical threshold
                robust_noise, echo_noise_array, valid_echoes = _robust_echo_noise(
                    X_centered,
                    z_thresh=echo_outlier_z
                )

                lambda_empirical = (
                    empirical_factor
                    * (robust_noise ** 2)
                    * (1.0 + np.sqrt(Ne / n_voxels)) ** 2
                )

                # ---- 3) combine theoretical / empirical
                # Report says weighted average according to coherence level.
                local_phase_coherence = float(
                    np.mean(
                        phase_coherence[ix1:ix2, jy1:jy2, kz1:kz2]
                    )
                )
                local_temporal_coherence = float(
                    np.mean(
                        temporal_coherence[ix1:ix2, jy1:jy2, kz1:kz2]
                    )
                )

                coherence = 0.5 * (
                    local_phase_coherence + local_temporal_coherence
                )

                base_threshold = (
                    coherence * lambda_theoretical
                    + (1.0 - coherence) * lambda_empirical
                )

                # ---- 4) coherence adjustment
                temporal_factor = (
                    1.0 - temporal_weight * local_temporal_coherence
                )

                phase_factor = (
                    1.0 - phase_weight * local_phase_coherence
                )

                if len(echo_noise_array) > 1:
                    cv_echo = (
                        np.std(echo_noise_array)
                        / (np.mean(echo_noise_array) + EPS)
                    )
                    consistency_factor = (
                        1.0
                        + consistency_weight * np.clip(cv_echo, 0, 1)
                    )
                else:
                    consistency_factor = 1.0

                # ---- 5) Rician-style correction
                rician_factor = 1.0

                if use_rician_correction:
                    mean_magnitude = np.sqrt(
                        np.mean(np.maximum(eigenvalues, 0.0))
                    )

                    snr_estimate = (
                        mean_magnitude / (robust_noise + EPS)
                    )

                    if snr_estimate > 3:
                        inside = max(
                            1.0 - 1.0 / (4.0 * snr_estimate ** 2),
                            0.0
                        )
                        rician_factor = np.sqrt(inside)
                    else:
                        rician_factor = 1.2

                final_threshold = (
                    base_threshold
                    * rician_factor
                    * temporal_factor
                    * phase_factor
                    * consistency_factor
                    * safety_margin
                )

                # ---- 6) signal component selection
                signal_mask = eigenvalues > final_threshold
                n_signal = int(np.sum(signal_mask))

                # report clamps to at least 1 signal component
                n_signal = max(1, min(n_signal, Ne))

                order = np.argsort(eigenvalues)[::-1]
                keep = order[:n_signal]

                V = Vh.conj().T
                W = V[:, keep]

                # Complex reconstruction
                X_denoised = (
                    X_centered
                    @ W
                    @ W.conj().T
                    + mean
                )

                X_denoised = X_denoised.reshape(*patch_size, Ne)

                # ---- 7) confidence-weighted aggregation
                confidence = coherence

                this_theta = (
                    (1.0 + confidence)
                    / (1.0 + Ne - n_signal)
                )

                theta[ix1:ix2, jy1:jy2, kz1:kz2] += this_theta
                thetax[ix1:ix2, jy1:jy2, kz1:kz2] += (
                    X_denoised * this_theta
                )

                threshold_map[i0, j0, k0] = final_threshold
                confidence_map[i0, j0, k0] = confidence
                nsignal_map[i0, j0, k0] = n_signal
                robust_noise_map[i0, j0, k0] = robust_noise

    denoised = np.zeros_like(thetax)

    valid = theta[..., 0] > 0
    denoised[valid] = thetax[valid] / theta[valid]
    denoised[~mask] = 0

    if not return_maps:
        return denoised.astype(out_dtype)

    maps = {
        "phase_coherence": phase_coherence,
        "temporal_coherence": temporal_coherence,
        "threshold_map": threshold_map,
        "confidence_map": confidence_map,
        "nsignal_map": nsignal_map,
        "robust_noise_map": robust_noise_map,
    }

    return denoised.astype(out_dtype), maps
