"""PROJECT_STATE.md next action 25 -- turns a pair of per-layer probe curves
(probe.py's linear, probe_mlp.py's nonlinear) into the specific verdict the
professor's question asks for, instead of leaving a human to eyeball two CSVs:

  "either this is a late-layer issue, in which case we have to work on
   saving the signal from earlier layers, or it is not captured by the
   encoder at all"

...plus the third option that framing leaves out, which next action 19 was
built specifically to catch: linearly inseparable but genuinely present
(needs a smarter reader, not a different layer).

Four verdicts, run per (task, encoder, manifest) so a task-varying answer
is visible rather than averaged away (Rupali's call, 2026-08-12: "maybe it
varies test to test"):

  NEVER_CAPTURED       neither linear nor nonlinear clears chance at ANY
                        layer -- not decodable by depth or by decoder choice
  LATE_LAYER_LOSS       clears chance in early/mid layers, drops back to
                        chance in the final layers -- fixable by reading an
                        earlier layer instead of the last one
  PRESENT_THROUGHOUT    clears chance from early layers on, no late decline
                        -- the representation isn't the bottleneck at all
  NONLINEAR_ONLY         linear never clears chance, nonlinear does somewhere
                        -- present, but a straight line can't reach it
  MIXED                 doesn't cleanly fit one of the above -- reported as
                        such, not forced into a bucket

Threshold: "clears chance" = acc >= chance + MARGIN (an absolute margin, not
a ratio -- a ratio breaks down at the very low chance rates several of these
tasks have, e.g. key_id's ~4%, where even noise can look like "2x chance").
MARGIN=0.08 was picked by checking it against numbers already reported
elsewhere in this project: mode_id's best own-encoder probe (0.04-0.12 vs
chance 0.077, PROJECT_STATE next action 13) stays BELOW this margin, matching
this project's own existing "barely above chance" characterization of that
number -- the threshold isn't invented in a vacuum, it reproduces a judgment
call this project already made by hand.

  python gpu/classify_layer_pattern.py \
      --linear results/trackB/probes/probe__acts_tag__task__target.csv \
      --nonlinear results/trackB/probes/probe_mlp__acts_tag__task__target.csv

UNVERIFIED against real curves (no real per-layer activations on this
laptop, same status as probe_mlp.py) -- verified against constructed
synthetic curves shaped like each of the four/five patterns, see
_selftest().
"""
import argparse

import numpy as np
import pandas as pd

MARGIN = 0.08
LATE_FRACTION = 1 / 6   # final sixth of layers counts as "late", matching how
                        # this project already talks about "final layers"
                        # elsewhere (e.g. attention_audio's early-vs-late reads)


def _above_chance(df: pd.DataFrame, acc_col: str) -> np.ndarray:
    return (df[acc_col] - df["chance"]) >= MARGIN


def classify(linear_df: pd.DataFrame, nonlinear_df: pd.DataFrame) -> dict:
    lin = linear_df.sort_values("layer").reset_index(drop=True)
    nl = nonlinear_df.sort_values("layer").reset_index(drop=True)
    n_layers = min(len(lin), len(nl))
    lin, nl = lin.iloc[:n_layers], nl.iloc[:n_layers]
    late_cut = max(1, int(round(n_layers * (1 - LATE_FRACTION))))

    lin_above = _above_chance(lin, "probe_acc").values
    nl_above = _above_chance(nl, "probe_acc_mlp").values
    any_above = lin_above | nl_above

    if not any_above.any():
        verdict = "NEVER_CAPTURED"
        note = (f"neither the linear nor the nonlinear decoder clears chance+{MARGIN} "
                f"at any of {n_layers} layers -- no evidence this property is decodable "
                "from this encoder's representation, at any depth or with either reader.")
    elif not any_above[:late_cut].any():
        # only ever clears chance in the late window -- not the "loss" story,
        # more like "only the late layers ever had it"; flag as MIXED rather
        # than force it into LATE_LAYER_LOSS, which specifically means EARLY
        # presence THEN loss.
        verdict = "MIXED"
        note = ("clears chance only in the final layers, never earlier -- doesn't match "
                "either 'late-layer loss' (needs early presence) or a clean absence.")
    elif any_above[late_cut:].any():
        verdict = "PRESENT_THROUGHOUT"
        note = (f"clears chance in early/mid layers AND stays above it through the final "
                f"{n_layers - late_cut} layers -- no late-layer decline; the representation "
                "isn't the bottleneck here.")
    elif lin_above[:late_cut].any():
        verdict = "LATE_LAYER_LOSS"
        peak_layer = int(lin.loc[:late_cut - 1, "probe_acc"].idxmax())
        note = (f"linear probe clears chance up to layer {peak_layer} (peak acc "
                f"{lin.loc[peak_layer, 'probe_acc']:.3f}) but drops back to chance by the "
                f"final {n_layers - late_cut} layers -- a readout built off layer "
                f"~{peak_layer} instead of the last layer should recover this, per the "
                "'save the signal from earlier layers' framing.")
    else:
        verdict = "NONLINEAR_ONLY"
        peak_layer = int(nl.loc[:late_cut - 1, "probe_acc_mlp"].idxmax())
        note = (f"linear probe never clears chance in early/mid layers, but the nonlinear "
                f"decoder does (peak layer {peak_layer}, acc "
                f"{nl.loc[peak_layer, 'probe_acc_mlp']:.3f}) -- present, but not linearly "
                "separable; a smarter readout fixes this, not a different layer.")
    return {"verdict": verdict, "note": note, "n_layers": n_layers, "late_cut": late_cut}


def main(linear_path: str, nonlinear_path: str):
    lin = pd.read_csv(linear_path)
    nl = pd.read_csv(nonlinear_path)
    result = classify(lin, nl)
    print(f"VERDICT: {result['verdict']}")
    print(result["note"])
    return result


def _mk(accs, chances):
    return pd.DataFrame({"layer": range(len(accs)), "probe_acc": accs, "chance": chances})


def _selftest():
    n, chance = 24, 0.08
    late = int(round(n * (1 - LATE_FRACTION)))

    # NEVER_CAPTURED: flat at chance, linear and nonlinear both
    flat = [chance + 0.01] * n
    r = classify(_mk(flat, [chance] * n),
                _mk(flat, [chance] * n).rename(columns={"probe_acc": "probe_acc_mlp"}))
    assert r["verdict"] == "NEVER_CAPTURED", r
    print("NEVER_CAPTURED: OK")

    # LATE_LAYER_LOSS: linear clear early, drops to chance in the final layers
    lin_curve = [chance + 0.20 if i < late else chance + 0.01 for i in range(n)]
    nl_curve = list(lin_curve)  # nonlinear tracks linear here (no extra info)
    r = classify(_mk(lin_curve, [chance] * n),
                _mk(nl_curve, [chance] * n).rename(columns={"probe_acc": "probe_acc_mlp"}))
    assert r["verdict"] == "LATE_LAYER_LOSS", r
    print("LATE_LAYER_LOSS: OK")

    # PRESENT_THROUGHOUT: clear the whole way
    full_curve = [chance + 0.25] * n
    r = classify(_mk(full_curve, [chance] * n),
                _mk(full_curve, [chance] * n).rename(columns={"probe_acc": "probe_acc_mlp"}))
    assert r["verdict"] == "PRESENT_THROUGHOUT", r
    print("PRESENT_THROUGHOUT: OK")

    # NONLINEAR_ONLY: linear flat everywhere, nonlinear clear early/mid
    lin_flat = [chance + 0.01] * n
    nl_clear = [chance + 0.20 if i < late else chance + 0.01 for i in range(n)]
    r = classify(_mk(lin_flat, [chance] * n),
                _mk(nl_clear, [chance] * n).rename(columns={"probe_acc": "probe_acc_mlp"}))
    assert r["verdict"] == "NONLINEAR_ONLY", r
    print("NONLINEAR_ONLY: OK")

    # MIXED: only clears chance in the LATE window, never earlier
    late_only = [chance + 0.01 if i < late else chance + 0.20 for i in range(n)]
    r = classify(_mk(late_only, [chance] * n),
                _mk(late_only, [chance] * n).rename(columns={"probe_acc": "probe_acc_mlp"}))
    assert r["verdict"] == "MIXED", r
    print("MIXED: OK")

    print("ALL CLASSIFIER PATTERNS VERIFIED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear")
    ap.add_argument("--nonlinear")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        if not (a.linear and a.nonlinear):
            raise SystemExit("--linear and --nonlinear required (or --selftest)")
        main(a.linear, a.nonlinear)
