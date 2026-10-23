"""analysis/noise_recompose.py: errors-in-variables correction of the one-parameter shrinkage gain at 10 ps, with a per-protein noise variance and a nested bootstrap.

Model: z = s + n for a stationary signal s and extraction noise n independent of s and between runs. Var = sigma_s^2 + sigma_n^2, nu = sigma_n^2 / Var; the fitted shrinkage coefficient is alpha = rho (1 - nu)
(rho: signal autocorrelation at the lag), and the shrinkage improvement over identity is g = (1 - alpha)/2 = (1 - rho)/2 + rho nu / 2. Given the observed g and nu: rho = (1 - 2g)/(1 - nu), noise-free g0 = (1 - rho)/2.
"""
from __future__ import annotations
import numpy as np
from fluxion.analysis.stats import entry_groups


def corrected_gain(g, nu):
    """g, nu: fractions (scalars or arrays). Returns the noise-free gain g0 = (1 - rho)/2 with rho = (1 - 2g)/(1 - nu)."""
    g, nu = np.asarray(g, float), np.asarray(nu, float)
    rho = (1.0 - 2.0 * g) / (1.0 - nu)
    return (1.0 - rho) / 2.0


def flags(g, nu):
    """Proteins where the correction is outside its domain: nu >= 0.5 or rho > 1 (not clipped, reported)."""
    g, nu = np.asarray(g, float), np.asarray(nu, float)
    rho = (1.0 - 2.0 * g) / (1.0 - nu)
    return (nu >= 0.5) | (rho > 1.0)


def recompose(proteins, var, g, sigma2_point, sigma2_draws, n_boot=10000, seed=0, return_diff=False):
    """Mean over proteins of the noise-free gain (percent) with a NESTED 95% bootstrap interval.

    var[p]: latent variance per element of the rows the gain was fit on; g[p]: observed shrinkage gain (fraction); sigma2_point[p]: noise variance per element, measured on that protein;
    sigma2_draws[p]: array of bootstrap draws of that measurement. Outer loop resamples PDB entries (all chains of a drawn entry come along); inner, each drawn protein's noise variance is redrawn
    from its own measurement distribution, so the measurement error of the noise variance is carried into the interval. Returns (estimate, lo, hi) in percent; with return_diff=True also the same
    triple for the paired difference (observed gain minus noise-free gain, the points attributable to noise) from the same resamples.
    """
    proteins = list(proteins); groups = entry_groups(proteins); entries = list(groups)
    nu = {p: sigma2_point[p] / var[p] for p in proteins}
    est = 100.0 * float(np.mean([corrected_gain(g[p], nu[p]) for p in proteins]))
    rng = np.random.default_rng(seed); boots = np.empty(n_boot); dboots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(entries), size=len(entries))
        vals, obs = [], []
        for i in pick:
            for p in groups[entries[i]]:
                s2 = float(sigma2_draws[p][rng.integers(0, len(sigma2_draws[p]))])
                vals.append(corrected_gain(g[p], s2 / var[p])); obs.append(g[p])
        boots[b] = 100.0 * float(np.mean(vals)); dboots[b] = 100.0 * (float(np.mean(obs)) - float(np.mean(vals)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if return_diff:
        dlo, dhi = np.percentile(dboots, [2.5, 97.5]); dest = 100.0 * float(np.mean([g[p] for p in proteins])) - est
        return (est, float(lo), float(hi)), (dest, float(dlo), float(dhi))
    return est, float(lo), float(hi)
