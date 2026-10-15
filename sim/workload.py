"""Closed loop single user workload. Request lengths are real (LMSYS Chat 1M, OLMoE
tokenizer); think time gaps are synthetic because the dataset has no timestamps."""
import os, numpy as np, pandas as pd
from config import DATA_DIR, PREFILL_MULT

_REQ = None
def requests_table():
    global _REQ
    if _REQ is None:
        path = os.path.join(DATA_DIR, "requests.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing. Run Step 2 of the notebook first.")
        _REQ = pd.read_csv(path)[["prompt_tokens", "response_tokens"]].to_numpy(float)
    return _REQ

def workload_stats(think_range):
    """Offline profile available to the controller: mean prompt, response and think time."""
    r = requests_table()
    return float(r[:, 0].mean()), float(r[:, 1].mean()), float(np.mean(think_range))

class Workload:
    def __init__(self, think_range, rng):
        req = requests_table()
        self.req = req[rng.permutation(len(req))]
        self.rng, self.think_range = rng, think_range
        self.i, self.t = 0, 0.0
        self.think_left = rng.uniform(*think_range)
        self.cur = None
        self.done = []          # (latency, ttft, decode_tps)

    def busy(self):
        return self.cur is not None or self.think_left <= 0

    def _start(self):
        pt, rt = self.req[self.i % len(self.req)]; self.i += 1
        self.cur = dict(arr=self.t, prefill=float(pt), decode=float(rt), rt=float(rt), first=None)

    def advance(self, decode_tps, dt=1.0):
        """Run one second. Returns (active fraction, decoded tokens)."""
        budget, active, decoded = dt, 0.0, 0.0
        while budget > 1e-9:
            if self.cur is None:
                if self.think_left > 0:
                    w = min(budget, self.think_left)
                    self.think_left -= w; budget -= w
                    if self.think_left > 1e-12: break
                    self.think_left = 0.0
                self._start(); self.cur["arr"] = self.t + (dt - budget)
            c = self.cur
            if c["prefill"] > 0:
                need = c["prefill"] / (decode_tps * PREFILL_MULT)
                w = min(budget, need)
                c["prefill"] -= w * decode_tps * PREFILL_MULT
                budget -= w; active += w
                if c["prefill"] <= 1e-9:
                    c["prefill"] = 0.0; c["first"] = self.t + (dt - budget)
                continue
            need = c["decode"] / decode_tps
            w = min(budget, need)
            c["decode"] -= w * decode_tps; decoded += w * decode_tps
            budget -= w; active += w
            if c["decode"] <= 1e-9:
                end = self.t + (dt - budget)
                dec_t = max(end - c["first"], 1e-6)
                self.done.append((end - c["arr"], c["first"] - c["arr"], c["rt"] / dec_t))
                self.cur = None
                self.think_left = self.rng.uniform(*self.think_range)
        self.t += dt
        return active / dt, decoded
