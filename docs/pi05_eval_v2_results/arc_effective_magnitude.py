"""What magnitude of perturbation did the arc augmentation actually apply?

PERTURB_ARC_STD=0.012 has been described throughout the project as "1.2cm perturbation". That
number is the *peak* offset, reached at a single frame in the middle of each subtask's free zone.
The envelope tapers to exactly zero (in value and slope) at both ends of that zone, and the last
freeze_frac of every subtask is left byte-identical to the source. Integrating the envelope gives
the magnitude the data actually carries, which is roughly a third of the peak.

This matters because the comparison that decides whether the augmentation could plausibly do
anything is against the variation already present in the demonstrations: the measured cross-demo
residual dispersion is 2.83cm (subtask 1) and 4.61cm (subtask 3), after removing object position.

Caveat on units: the arc number below is a mean offset magnitude while the natural-variation
numbers are dispersions, so the ratios are order-of-magnitude comparisons, not exact fractions.
An order of magnitude is not sensitive to that distinction.

Mirrors _apply_arc_perturbation in isaaclab_mimic/datagen/data_generator.py (order-6 family,
peak_frac=0.5, which is what production used).
"""

import numpy as np

ORDER = 6.0
FREEZE_FRAC = 0.3  # production default; the trailing 30% of each subtask is untouched
NATURAL = {
    "subtask1 cross-demo residual dispersion": 2.83,
    "subtask3 cross-demo residual dispersion": 4.61,
    "subtask1 within-path natural curvature": 4.79,
    "subtask3 within-path natural curvature": 9.33,
}


def envelope_mean(peak_frac: float = 0.5, n: int = 2000) -> float:
    a, b = ORDER * peak_frac, ORDER * (1.0 - peak_frac)
    peak_value = (peak_frac**a) * ((1.0 - peak_frac) ** b)
    u = np.clip(np.linspace(0.0, 1.0, n), 1e-6, 1.0 - 1e-6)
    return float((((u**a) * ((1.0 - u) ** b)) / peak_value).mean())


def main():
    m = envelope_mean()
    free = 1.0 - FREEZE_FRAC
    print(f"envelope mean over the free zone      = {m:.4f}")
    print(f"free zone as a fraction of a subtask  = {free:.2f}  (freeze_frac={FREEZE_FRAC})")
    print(f"=> mean displacement = peak * {m:.4f} * {free:.2f} = peak * {m * free:.4f}\n")

    for label, peak in [("production (v2 delivery)", 1.2), ("largest point in the v2 sweep", 3.0)]:
        mean_disp = peak * m * free
        print(f"{label}: peak {peak:.1f}cm -> mean displacement {mean_disp:.2f}cm")
        for name, nat in NATURAL.items():
            print(f"    vs {name:42s} {nat:5.2f}cm -> {mean_disp / nat * 100:5.1f}%")
        print()

    print("peak needed to make the mean displacement match natural variation:")
    for name, nat in NATURAL.items():
        print(f"    {name:42s} {nat:5.2f}cm -> peak {nat / (m * free):5.2f}cm")
    print(
        "\nThe v2 sweep already measured generation success falling to 19.1% at a 3.0cm peak, and it"
        "\nwas still dropping there. The peaks above sit far beyond that, so matching the data's own"
        "\nvariation costs generation yield -- that trade-off is itself the failure boundary the"
        "\ndesign document set out to characterize."
    )


if __name__ == "__main__":
    main()
