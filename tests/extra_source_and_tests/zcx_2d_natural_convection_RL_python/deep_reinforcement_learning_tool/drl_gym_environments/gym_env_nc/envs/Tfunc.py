import sympy
import numpy as np


class Tfunc:
    """
    Generate a non-periodic bottom-wall temperature profile:
    - The interval [x0, x1] is uniformly divided into `nb_seg` segments.
    - Each segment takes a plateau value T_k (derived from dicTemp["Tk"] with amplitude limiting and mean-centering).
    - Near segment interfaces, we use cubic polynomials over a band of width `dx` to ensure C¹ continuity. in CPP
    - The first segment has no left transition; the last segment has no right transition (no wrap-around).
    """

    def __init__(self, nb_seg: int, dicTemp: dict,
                 x_interval=(0.0, 2*sympy.pi), ampl=0.75, dx=0.03):
        """
        nb_seg    : number of segments (actuators)
        dicTemp   : e.g., {"T0": v0, "T1": v1, ...}; typically each v ∈ [-1, 1]
        x_interval: bottom-wall coordinate range (x0, x1)
        ampl      : max amplitude (final temperature range ≈ 2 ± ampl)
        dx        : half-width of the smoothing band near each interface
        """
        self.nb_seg = int(nb_seg)
        self.dicTemp = dict(dicTemp)
        self.x0, self.x1 = map(float, x_interval)
        self.ampl = float(ampl)
        self.dx = float(dx)

        if self.nb_seg < 1:
            raise ValueError("nb_seg must be ≥ 1")
        if self.x1 <= self.x0:
            raise ValueError("x_interval must satisfy x1 > x0")

    def apply_T(self, x):
        """
        Return a sympy.Piecewise(x) representing the temperature profile T(x).
        """
        # 1) Amplitude limiting and mean-centering:
        #    map dicTemp to 2 ± ampl and apply a global scale to avoid exceeding ±ampl.
        values = self.ampl * np.array([self.dicTemp.get(f"T{i}", 0.0) for i in range(self.nb_seg)], dtype=float)
        Mean = float(values.mean())
        K2 = max(1.0, float(np.abs(values - Mean).max() / self.ampl)) if self.ampl > 0 else 1.0

        T = [2.0 + (self.ampl * self.dicTemp.get(f"T{i}", 0.0) - Mean) / K2 for i in range(self.nb_seg)]

        # 2) Geometry and effective smoothing width
        L = self.x1 - self.x0
        seg_len = L / self.nb_seg
        x_edges = [self.x0 + i * seg_len for i in range(self.nb_seg + 1)]
        pieces = []
        for k in range(self.nb_seg):
            xk = x_edges[k]
            xk1 = x_edges[k + 1]
            Tk = T[k]

            if k < self.nb_seg - 1:
                # normal segments: x < xk1
                pieces.append((Tk, x < xk1))
            else:
                # last one: catch-all True to extend to x1
                pieces.append((Tk, True))

        return sympy.Piecewise(*pieces)



        # limit dx to at most half a segment (avoid overlap of left/right bands)
        # dx_eff = min(self.dx, 0.5 * seg_len * 0.98)
        #
        # # segment edges
        # x_edges = [self.x0 + i * seg_len for i in range(self.nb_seg + 1)]
        #
        # seq = []  # list of (expr, condition) for Piecewise
        #
        # # 3) Build Piecewise by segments
        # for k in range(self.nb_seg):
        #     xk = x_edges[k]
        #     xk1 = x_edges[k + 1]
        #     Tk = T[k]
        #
        #     left_has_transition = (k > 0)
        #     right_has_transition = (k < self.nb_seg - 1)
        #
        #     # (a) Left transition: T_{k-1} → T_k over [xk, xk+dx_eff]
        #     if left_has_transition and dx_eff > 0:
        #         T_left = T[k - 1]
        #         expr_left = T_left + ((T_left - Tk) / (4 * dx_eff ** 3)) * (x - xk - 2 * dx_eff) * (
        #                     x - xk + dx_eff) ** 2
        #         seq.append((expr_left, x < xk + dx_eff))
        #
        #     # (b) Plateau region: [xk + (left?dx_eff:0), xk1 - (right?dx_eff:0)]
        #     mid_left = xk + (dx_eff if left_has_transition and dx_eff > 0 else 0.0)
        #     mid_right = xk1 - (dx_eff if right_has_transition and dx_eff > 0 else 0.0)
        #     if mid_right > mid_left:
        #         seq.append((Tk, x < mid_right))
        #
        #     # (c) Right transition: T_k → T_{k+1} over [xk1 - dx_eff, xk1]
        #     if right_has_transition and dx_eff > 0:
        #         T_right = T[k + 1]
        #         expr_right = Tk + ((Tk - T_right) / (4 * dx_eff ** 3)) * (x - xk1 - 2 * dx_eff) * (
        #                     x - xk1 + dx_eff) ** 2
        #         seq.append((expr_right, x < xk1))
        #
        #     # (d) Final fallback for the last segment
        #     if k == self.nb_seg - 1:
        #         # If no right transition, keep plateau Tk to the end.
        #         seq.append((Tk, True))
        #
        # return sympy.Piecewise(*seq)
