"""RLC compact-model extraction and circulax inverse design.

This module provides three stages of inductor model extraction:

1. **Analytical initialisation** – ``estimate_rlc_params`` reads f0, Q, R
   directly from the |Z| data (peak location, DC resistance, -3 dB bandwidth).

2. **JAX / Optax fitting** – ``fit_rlc`` minimises the complex squared error
   between ``R · z_rlc(f/f0, Q)`` and the target impedance using Adam.

3. **Circulax inverse design** – ``circulax_refine`` compiles a one-port RLC
   netlist with :func:`circulax.compile_netlist`, sweeps S-parameters via
   :func:`circulax.setup_ac_sweep`, and refines R, L, C with Adam using JAX
   automatic differentiation through the full circuit simulator.

Public API
----------
z_rlc              – normalised parallel-RLC impedance (JAX-traceable)
estimate_rlc_params – analytical initial guess from |Z| data
fit_rlc             – JAX / Optax optimisation returning (f0, Q, R)
recover_LC          – compute L and C from (f0, Q, R)
circulax_refine     – differentiable inverse design with circulax
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

from circulax import compile_netlist, fdomain_component, setup_ac_sweep
from circulax.utils import update_params_dict


# ── Normalised parallel-RLC impedance ─────────────────────────────────────────

def z_rlc(w: jnp.ndarray, Q: float) -> jnp.ndarray:
    """Normalised parallel-RLC impedance.

    The circuit is a series RL branch in parallel with capacitor C::

             --- C ---
            |         |
        p1 -+---R--L--+- p2

    Parameters
    ----------
    w:
        Dimensionless frequency  ``f / f0``.
    Q:
        Quality factor  ``Q = ω₀ L / R``.

    Returns
    -------
    complex array with the same shape as *w*::

        z_norm = Z / R = (1 + j·w·Q) / (1 − w² + j·w/Q)
    """
    return (1 + 1j * w * Q) / (1 - w**2 + 1j * w / Q)


# ── Analytical initial-guess ──────────────────────────────────────────────────

def estimate_rlc_params(
    f_hz: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate initial RLC parameters from impedance magnitude data.

    Parameters
    ----------
    f_hz:
        Frequencies in Hz, shape ``(N,)``.
    Z:
        Complex impedance values, shape ``(N,)``.

    Returns
    -------
    f0_ini : float
        Resonant frequency [Hz] – frequency at which |Z| is maximum.
    Q_ini : float
        Quality factor estimated from ``sqrt(|Z_peak| / |R_ini|)``.
    R_ini : float
        DC resistance [Ω] – real part of Z at the lowest frequency.
    """
    absZ    = np.abs(Z)
    idx_pk  = int(np.argmax(absZ))
    f0_ini  = float(f_hz[idx_pk])
    R_ini   = float(Z.real[0])
    # Q from Im(Z)/Re(Z) at the |Z| peak – the physical inductor Q at resonance.
    # This is a better seed than bandwidth methods when the resonance is broad.
    Re_pk = float(Z[idx_pk].real)
    Im_pk = float(Z[idx_pk].imag)
    Q_ini = abs(Im_pk / Re_pk) if abs(Re_pk) > 1e-15 else 5.0
    return f0_ini, Q_ini, R_ini


# ── JAX / Optax RLC fitting ───────────────────────────────────────────────────

def fit_rlc(
    f_hz: np.ndarray,
    Z: np.ndarray,
    *,
    steps: int = 1000,
    lr: float = 0.05,
    verbose: bool = True,
    log_every: int = 200,
) -> tuple[float, float, float]:
    """Fit an RLC model to complex impedance data using JAX and Optax Adam.

    Minimises the real-valued loss::

        L(f0, Q, R) = Σ_k |Z_target(fk) − R · z_rlc(fk/f0, Q)|²

    Parameters are encoded in log-space (``log f0``, ``log Q``, ``log R``) so
    that positivity is guaranteed throughout optimisation.

    Parameters
    ----------
    f_hz:
        Frequencies in Hz, shape ``(N,)``.
    Z:
        Target complex impedance, shape ``(N,)``.
    steps:
        Number of Adam iterations.
    lr:
        Adam learning rate.
    verbose:
        Print progress every *log_every* steps.
    log_every:
        Logging interval (steps).

    Returns
    -------
    f0_fit, Q_fit, R_fit : float
        Fitted resonant frequency [Hz], quality factor, and resistance [Ω].
    """
    f0_ini, Q_ini, R_ini = estimate_rlc_params(f_hz, Z)
    if verbose:
        print(
            f"  Initial guess:  f0 = {f0_ini / 1e9:.3f} GHz  "
            f"Q = {Q_ini:.3f}  R = {R_ini:.4f} Ω"
        )

    f_jnp    = jnp.array(f_hz, dtype=jnp.float64)
    Z_target = jnp.array(Z,    dtype=jnp.complex128)

    # Log-space encoding ensures f0, Q, R > 0 at all times
    def loss_fn(log_param):
        f0 = jnp.exp(log_param[0])
        Q  = jnp.exp(log_param[1])
        R  = jnp.exp(log_param[2])
        z_fit = R * z_rlc(f_jnp / f0, Q)
        z_err = Z_target - z_fit
        return jnp.real(jnp.sum(z_err * jnp.conj(z_err)))

    log_par   = jnp.array([np.log(f0_ini), np.log(Q_ini), np.log(R_ini)])
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(log_par)
    vg_fn     = jax.jit(jax.value_and_grad(loss_fn))

    for step in range(steps):
        loss, grads = vg_fn(log_par)
        if verbose and step % log_every == 0:
            f0_, Q_, R_ = [float(np.exp(v)) for v in log_par]
            print(
                f"  step {step:4d}:  f0 = {f0_ / 1e9:.4f} GHz  "
                f"Q = {Q_:.3f}  R = {R_:.4f}  "
                f"loss = {float(loss):.3e}"
            )
        updates, opt_state = optimizer.update(grads, opt_state)
        log_par = optax.apply_updates(log_par, updates)

    f0_fit, Q_fit, R_fit = [float(np.exp(v)) for v in log_par]
    return f0_fit, Q_fit, R_fit


# ── Recover L and C from fitted RLC parameters ────────────────────────────────

def recover_LC(
    f0_fit: float,
    Q_fit:  float,
    R_fit:  float,
) -> tuple[float, float]:
    """Recover inductance L and capacitance C from fitted RLC parameters.

    Relations::

        ω₀ = 2π f0,   τ = Q / ω₀,   L = τ · R,   C = 1 / (L · ω₀²)

    Parameters
    ----------
    f0_fit:
        Resonant frequency [Hz].
    Q_fit:
        Quality factor.
    R_fit:
        Series resistance [Ω].

    Returns
    -------
    L [H], C [F]
    """
    w0    = 2.0 * np.pi * f0_fit
    tau   = Q_fit / w0
    L_fit = tau * R_fit
    C_fit = 1.0 / (L_fit * w0 ** 2)
    return L_fit, C_fit


# ── Circulax frequency-domain component (defined once at module level) ────────

@fdomain_component(ports=("p1", "p2"))
def _RLCInductor(f, R=1.0, L=100e-12, C=10e-15):
    """Two-port admittance of a parallel-RL+C one-port inductor model.

    Circuit::

             --- C ---
            |         |
        p1 -+---R--L--+- p2
    """
    w    = 2.0 * jnp.pi * f
    Y_RL = 1.0 / (R + 1j * w * L)
    Y_C  = 1j * w * C
    Y    = Y_RL + Y_C
    return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)


# ── Circulax inverse design ────────────────────────────────────────────────────

def circulax_refine(
    f_hz:  np.ndarray,
    Z:     np.ndarray,
    R_ini: float,
    L_ini: float,
    C_ini: float,
    *,
    steps:     int   = 200,
    lr:        float = 1e-2,
    verbose:   bool  = True,
    log_every: int   = 20,
) -> tuple[float, float, float]:
    """Refine RLC parameters using a circulax differentiable inverse-design loop.

    Compiles a one-port ``R–L ∥ C`` netlist with :func:`circulax.compile_netlist`,
    performs an S-parameter AC sweep via :func:`circulax.setup_ac_sweep`, and
    minimises the mean squared error::

        L(R, L, C) = mean_f |Z_cx(f; R,L,C) − Z_target(f)|²

    where ``Z_cx`` is recovered from the circulax S11 as::

        Z_cx = Z0 · (1 + S11) / (1 − S11),   Z0 = 50 Ω

    A softplus reparametrisation ``R = softplus(ρ)`` etc. keeps the parameters
    strictly positive throughout optimisation.

    Parameters
    ----------
    f_hz:
        Frequencies in Hz, shape ``(N,)``.
    Z:
        Target complex impedance, shape ``(N,)``.
    R_ini, L_ini, C_ini:
        Initial RLC values (e.g. from :func:`fit_rlc` + :func:`recover_LC`).
    steps:
        Adam iterations.
    lr:
        Adam learning rate.
    verbose:
        Print progress every *log_every* steps.
    log_every:
        Logging interval (steps).

    Returns
    -------
    R_fit, L_fit, C_fit : float
        Refined resistance [Ω], inductance [H], and capacitance [F].
    """
    # ── Compile the one-port netlist once ────────────────────────────────────
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "L1": {
                "component": "rlc_inductor",
                "settings": {"R": R_ini, "L": L_ini, "C": C_ini},
            },
        },
        "connections": {
            "L1,p1": "IN",
            "L1,p2": "GND,p1",
        },
    }
    models = {"rlc_inductor": _RLCInductor, "ground": lambda: 0}

    groups, sys_size, port_map = compile_netlist(net_dict, models)
    port_node = port_map["IN"]
    y_dc      = jnp.zeros(sys_size)

    freqs    = jnp.asarray(f_hz, dtype=jnp.float64)
    Z_target = jnp.asarray(Z,    dtype=jnp.complex128)

    # ── Loss: circulax AC sweep → S11 → Z → MSE ──────────────────────────────
    def loss_fn(raw_params: jnp.ndarray) -> jnp.ndarray:
        R = jnp.exp(raw_params[0])
        L = jnp.exp(raw_params[1])
        C = jnp.exp(raw_params[2])
        g = update_params_dict(groups, "rlc_inductor", "L1", "R", R)
        g = update_params_dict(g,      "rlc_inductor", "L1", "L", L)
        g = update_params_dict(g,      "rlc_inductor", "L1", "C", C)
        ac  = setup_ac_sweep(groups=g, num_vars=sys_size, port_nodes=[port_node])
        S   = ac(y_dc, freqs)           # (N_freqs, 1, 1)
        S11 = S[:, 0, 0]
        Z_cx   = 50.0 * (1 + S11) / (1 - S11)
        err_re = jnp.real(Z_cx) - jnp.real(Z_target)
        err_im = jnp.imag(Z_cx) - jnp.imag(Z_target)
        return jnp.mean(err_re ** 2 + err_im ** 2)

    # ── Initialise optimiser in log-space (ensures R, L, C > 0) ──────────────
    # Scale to convenient magnitudes before taking log so that all log-params
    # are O(1) and Adam steps are uniform across orders of magnitude.
    raw_ini = jnp.array([
        np.log(R_ini),
        np.log(L_ini),
        np.log(C_ini),
    ])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(raw_ini)
    vg_fn     = jax.jit(jax.value_and_grad(loss_fn))
    vg_fn(raw_ini)   # warm-up JIT compilation

    # ── Adam loop ─────────────────────────────────────────────────────────────
    raw = raw_ini
    for step in range(steps):
        loss, grads = vg_fn(raw)
        if verbose and step % log_every == 0:
            R_, L_, C_ = [float(np.exp(v)) for v in raw]
            print(
                f"  step {step:3d}:  R = {R_:.5f} Ω  "
                f"L = {L_ * 1e12:.3f} pH  C = {C_ * 1e15:.3f} fF  "
                f"loss = {float(loss):.3e}"
            )
        updates, opt_state = optimizer.update(grads, opt_state)
        raw = optax.apply_updates(raw, updates)

    R_fit, L_fit, C_fit = [float(np.exp(v)) for v in raw]
    return R_fit, L_fit, C_fit
