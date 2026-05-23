"""
Fed-batch penicillin fermentation simulator.

Implements simplified Monod + Luedeking-Piret kinetics inspired by the
IndPenSim benchmark (Goldrick et al. 2015, 2019). Generates realistic
batch-to-batch variability through stochastic parameter sampling and
operator-controlled feed profiles.

This is a teaching/portfolio simulator, NOT a substitute for IndPenSim.
For the canonical fermentation benchmark dataset, see:
  Goldrick S. et al. (2019) "Modern day monitoring and control challenges
  outlined on an industrial-scale benchmark fermentation process."
  Computers & Chemical Engineering 130:106471.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Biological constants — central tendencies from the literature
# (Goldrick et al. 2015; Birol et al. 2002; standard P. chrysogenum values)
# --------------------------------------------------------------------------
@dataclass
class FermentationParams:
    """Mean kinetic parameters; per-batch values are sampled around these."""

    # Growth kinetics
    mu_max: float = 0.10  # max specific growth rate (1/h)
    K_S: float = 0.15  # substrate half-saturation (g/L)
    k_d: float = 0.005  # cell death rate (1/h)

    # Substrate consumption
    Y_XS: float = 0.45  # biomass yield on substrate (g X / g S)
    m_S: float = 0.014  # maintenance coefficient (1/h)

    # Product formation (Luedeking-Piret)
    alpha: float = 0.10  # growth-associated coefficient
    beta: float = 0.005  # non-growth-associated coefficient (1/h)
    k_P: float = 0.001  # product degradation (1/h)

    # Reactor
    V0: float = 100.0  # initial volume (L)
    S_feed: float = 500.0  # sugar feed concentration (g/L)
    X0: float = 0.10  # initial biomass (g/L)
    S0: float = 1.0  # initial substrate (g/L)
    P0: float = 0.0  # initial product (g/L)

    # Environmental setpoints
    T_setpoint: float = 298.0  # K (≈ 25 °C)
    pH_setpoint: float = 6.5
    DO_setpoint: float = 60.0  # % saturation


# --------------------------------------------------------------------------
# Per-batch stochastic variation
# --------------------------------------------------------------------------
def sample_batch_params(
    base: FermentationParams,
    rng: np.random.Generator,
    cv: float = 0.06,
) -> FermentationParams:
    """Draw a per-batch parameter set by perturbing the central values.

    cv = coefficient of variation (std/mean) applied independently to each
    kinetic parameter. 10% mimics realistic biological batch variability.
    """
    perturbed = FermentationParams(**base.__dict__)
    for key in ["mu_max", "K_S", "k_d", "Y_XS", "alpha", "beta"]:
        mean = getattr(base, key)
        perturbed.__dict__[key] = mean * rng.lognormal(mean=0.0, sigma=cv)
    return perturbed


# --------------------------------------------------------------------------
# Feed profile — operator-controlled sugar feed flow
# --------------------------------------------------------------------------
def generate_feed_profile(
    duration_h: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return F_feed(t) in L/h. Operator strategies vary batch-to-batch.

    Three style families, sampled uniformly:
      - 'gradual': slow linear ramp up over the first half of the batch
      - 'aggressive': fast ramp early, then constant
      - 'pulsed': sequence of step increases
    """
    n = int(duration_h / dt) + 1
    t = np.arange(n) * dt
    profile = np.zeros(n)

    style = rng.choice(["gradual", "aggressive"])
    F_max = rng.uniform(0.07, 0.09)  # L/h, peak feed rate

    if style == "gradual":
        ramp_end = duration_h * rng.uniform(0.4, 0.6)
        profile = np.minimum(t / ramp_end, 1.0) * F_max
    elif style == "aggressive":
        ramp_end = duration_h * rng.uniform(0.1, 0.2)
        profile = np.minimum(t / ramp_end, 1.0) * F_max
    else:  # pulsed
        n_steps = rng.integers(3, 7)
        step_times = np.sort(rng.uniform(0, duration_h, n_steps))
        step_heights = np.cumsum(rng.uniform(0.01, 0.025, n_steps))
        step_heights = np.minimum(step_heights, F_max)
        for st_time, st_h in zip(step_times, step_heights):
            profile = np.where(t >= st_time, st_h, profile)

    # Feed starts only after a lag phase (no sugar in early hours)
    lag = rng.uniform(8, 15)
    profile = np.where(t < lag, 0.0, profile)
    return profile


# --------------------------------------------------------------------------
# ODE integration — explicit Euler is fine at this timescale
# --------------------------------------------------------------------------
def integrate_batch(
    params: FermentationParams,
    feed_profile: np.ndarray,
    duration_h: float,
    dt: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Integrate the three-state ODE system through one batch.

    State variables: X (biomass), S (substrate), P (product), V (volume).
    """
    n = int(duration_h / dt) + 1
    t = np.arange(n) * dt

    X = np.zeros(n)
    X[0] = params.X0
    S = np.zeros(n)
    S[0] = params.S0
    P = np.zeros(n)
    P[0] = params.P0
    V = np.zeros(n)
    V[0] = params.V0

    # Environmental signals — these are CONTROLLED by the regulator so they
    # mostly stay near setpoint, with occasional disturbances
    T = np.full(n, params.T_setpoint) + rng.normal(0, 0.3, n)
    pH = np.full(n, params.pH_setpoint) + rng.normal(0, 0.05, n)
    DO = np.full(n, params.DO_setpoint) + rng.normal(0, 2.0, n)

    # Inject 1–3 process disturbances at random times (temperature excursion,
    # pH drift, etc) — this is what creates the realistic "things go wrong"
    n_disturbances = rng.integers(0, 3)
    for _ in range(n_disturbances):
        center = rng.uniform(0.2, 0.9) * n
        width = rng.integers(20, 80)
        affected_var = rng.choice(["T", "pH", "DO"])
        idx = np.arange(max(0, int(center - width)), min(n, int(center + width)))
        bump = rng.uniform(-1, 1) * np.exp(-(((idx - center) / (width / 2)) ** 2))
        if affected_var == "T":
            T[idx] += bump * 0.7
        elif affected_var == "pH":
            pH[idx] += bump * 0.15
        else:
            DO[idx] += bump * 4.0

    # Agitator RPM — usually constant, with occasional operator adjustments
    RPM = np.full(n, 100.0) + rng.normal(0, 1.5, n)
    n_rpm_changes = rng.integers(0, 4)
    for _ in range(n_rpm_changes):
        change_idx = rng.integers(0, n)
        new_setpoint = rng.uniform(90, 120)
        RPM[change_idx:] = new_setpoint + rng.normal(0, 1.5, n - change_idx)

    # Environmental modulation of growth rate:
    # cells grow best near setpoint; deviations slow them down
    def env_factor(T_t, pH_t, DO_t):
        f_T = np.exp(-(((T_t - params.T_setpoint) / 5.0) ** 2))
        f_pH = np.exp(-(((pH_t - params.pH_setpoint) / 0.5) ** 2))
        f_DO = DO_t / (params.DO_setpoint + DO_t) if DO_t > 0 else 0.0
        return f_T * f_pH * f_DO

    # Integrate
    for i in range(n - 1):
        # Effective growth rate, modulated by environment
        mu = params.mu_max * S[i] / (params.K_S + S[i]) * env_factor(T[i], pH[i], DO[i])

        # Mass-balance derivatives (per unit volume)
        F = feed_profile[i]
        dilution = F / V[i] if V[i] > 0 else 0.0
        dXdt = (mu - params.k_d) * X[i] - dilution * X[i]
        dSdt = -(mu / params.Y_XS + params.m_S) * X[i] + dilution * (
            params.S_feed - S[i]
        )
        dPdt = (
            (params.alpha * mu + params.beta) * X[i]
            - params.k_P * P[i]
            - dilution * P[i]
        )
        dVdt = F

        # Euler step
        X[i + 1] = max(0.0, X[i] + dt * dXdt)
        S[i + 1] = max(0.0, S[i] + dt * dSdt)
        P[i + 1] = max(0.0, P[i] + dt * dPdt)
        V[i + 1] = V[i] + dt * dVdt

    return {
        "time_h": t,
        "biomass_gL": X,
        "substrate_gL": S,
        "penicillin_gL": P,
        "volume_L": V,
        "feed_rate_Lph": feed_profile,
        "temperature_K": T,
        "pH": pH,
        "DO_pct": DO,
        "agitator_rpm": RPM,
    }


# --------------------------------------------------------------------------
# Add sensor noise (online = clean-ish; offline = noisier and sparser)
# --------------------------------------------------------------------------
def add_sensor_noise(
    batch: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Add realistic measurement noise. Noise scales with signal magnitude
    (multiplicative + small additive floor), which mimics real biosensor behavior:
    relative error stays roughly constant across the measurement range.

    Online (DO, pH, T, RPM) handled in integrate_batch; this function handles
    the biological measurements (biomass, substrate, penicillin).
    """
    out = {k: v.copy() for k, v in batch.items()}

    def noisy(signal, rel_sigma, abs_floor):
        # multiplicative noise + small additive floor
        return signal * (1 + rng.normal(0, rel_sigma, len(signal))) + rng.normal(
            0, abs_floor, len(signal)
        )

    out["biomass_gL"] = noisy(batch["biomass_gL"], rel_sigma=0.05, abs_floor=0.05)
    out["substrate_gL"] = noisy(batch["substrate_gL"], rel_sigma=0.05, abs_floor=0.05)
    out["penicillin_gL"] = noisy(batch["penicillin_gL"], rel_sigma=0.05, abs_floor=0.02)

    out["biomass_gL"] = np.maximum(0.0, out["biomass_gL"])
    out["substrate_gL"] = np.maximum(0.0, out["substrate_gL"])
    out["penicillin_gL"] = np.maximum(0.0, out["penicillin_gL"])
    return out


# --------------------------------------------------------------------------
# Top-level: simulate a batch, simulate a campaign
# --------------------------------------------------------------------------
def simulate_batch(
    batch_id: int,
    duration_h: float = 200.0,
    dt: float = 0.5,
    seed: Optional[int] = None,
    base_params: Optional[FermentationParams] = None,
) -> pd.DataFrame:
    """Simulate a single fermentation batch; return a tidy DataFrame."""
    rng = np.random.default_rng(seed)
    base = base_params or FermentationParams()
    params = sample_batch_params(base, rng)
    feed = generate_feed_profile(duration_h, dt, rng)
    raw = integrate_batch(params, feed, duration_h, dt, rng)
    noisy = add_sensor_noise(raw, rng)

    df = pd.DataFrame(noisy)
    df.insert(0, "batch_id", batch_id)
    return df


def simulate_campaign(
    n_batches: int = 40,
    duration_h: float = 200.0,
    dt: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a campaign of n_batches and return one long DataFrame."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, n_batches)
    batches = [
        simulate_batch(i + 1, duration_h, dt, int(s)) for i, s in enumerate(seeds)
    ]
    return pd.concat(batches, ignore_index=True)
