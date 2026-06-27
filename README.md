# Helios — AI Solar Flare Predictor

A live, self-training solar-flare forecasting system. It learns from the
**SWAN-SF** magnetic-field dataset, classifies what the Sun is doing **right
now** from live NOAA data, and forecasts flare probability at **12 / 24 / 48 h**
lead times — with an escalating warning for X-class events. Ships with a
polished web **forecast page**.

> **First-model status:** the bundled smoke-test model reaches **TSS ≈ 0.77**
> on held-out data. On the real SWAN-SF M-class task expect TSS in the
> ~0.5–0.75 band — solidly above the "no-skill" line and improvable with more
> data/features. See [Honest limitations](#honest-limitations).

---

## The one thing you must understand first

**The training data and the live data are different physics**, and the system
is built to respect that instead of faking a connection:

| | Trains on | Predicts from (live) |
|---|---|---|
| **SHARP ML model** | SDO/HMI **magnetic** parameters (24 SHARP features) | needs live SHARP (JSOC) — optional |
| **Flux nowcast/forecast** | — (statistical) | NOAA **GOES X-ray flux** |
| **NOAA region forecast** | — (NOAA's own model) | NOAA region probabilities |

A model trained on magnetic SHARP parameters **cannot** literally consume X-ray
flux — they are different feature spaces. Pretending it could would be exactly
the "cheating" the project brief warns against. So the predictor runs **three
independent tracks** and ensembles whatever is available, always returning a
clean forecast even if a source is down.

```
                       ┌─────────────────────────────┐
  NOAA GOES X-ray ───► │ NOWCAST  (exact, no ML)      │ ─► current class + X-WARNING
   (always live)       └─────────────────────────────┘
                       ┌─────────────────────────────┐
                  ───► │ FLUX forecast (persistence) │ ─┐
                       └─────────────────────────────┘  │
  NOAA regions    ───► │ NOAA official region forecast│ ─┼─► ENSEMBLE ─► 12/24/48h
                       ┌─────────────────────────────┐  │
  SWAN-SF model   ───► │ SHARP ML  (when fed magnetic)│ ─┘
                       └─────────────────────────────┘
```

Because a flare's class is *defined* by its GOES 1–8 Å peak flux, identifying a
flare **happening now** needs no ML at all — it's a measurement. The machine
learning is reserved for **forecasting the future**, which is the honest place
for it.

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. prove the whole pipeline works (NO download, NO GPU needed)
python scripts/smoke_test.py
#    -> trains on a synthetic fixture, evaluates skill scores,
#       saves models/flare_sharp_model.pkl, then hits live NOAA.

# 3. run the API + open the forecast page
python -m uvicorn api.server:app --port 8000
#    -> open http://127.0.0.1:8000/
```

### Train on the real SWAN-SF data

```bash
python -m solarflare.download      # pulls the .pkl partitions (Google Drive)
python -m solarflare.train         # trains on partitions 1-3, tests on 5
```

Training writes three artifacts to `models/`:

| File | Use |
|---|---|
| `flare_sharp_model.joblib` | **load this** (recommended) |
| `flare_sharp_model.pkl` | same payload, plain pickle (you asked for `.pkl`) |
| `flare_sharp_model.meta.json` | human-readable metrics + provenance |

---

## How it avoids "cheating"

* **Leakage-free split** — trains on the augmented partitions, tunes the
  decision threshold on a *separate* validation partition, and reports final
  numbers on a **clean test partition the model never saw**. This is the
  standard SWAN-SF protocol.
* **Honest metrics** — accuracy is useless when 95% of samples are "no-flare",
  so the headline is **TSS** (True Skill Statistic = recall − false-alarm rate;
  0 = no skill) and **HSS** (Heidke Skill Score). See `solarflare/evaluate.py`.
* **Class imbalance** handled with balanced sample weights, not by resampling
  the test set.
* **Calibrated probabilities** — isotonic calibration so a "21%" forecast means
  roughly 21%.

---

## Flare taxonomy (matches the brief)

Driven entirely by `config.yaml`:

| Category | GOES 1–8 Å peak flux | Treated as |
|---|---|---|
| A, B, **weak C** (< C5) | < 5×10⁻⁶ W/m² | **no-flare** |
| **C** (≥ C5) | 5×10⁻⁶ – 10⁻⁵ | tracked |
| **M** | 10⁻⁵ – 10⁻⁴ | tracked |
| **X** | ≥ 10⁻⁴ | tracked + **WARNING** |
| X ≥ X5 / X10 | ≥ 5×10⁻⁴ / 10⁻³ | **SEVERE / EXTREME** |

The default trainable target is **"any M-or-greater flare in the next 24 h"**
(the Cleaned-SWANSF labels are binary). To get a 4-way *no-flare/C/M/X* model,
supply the **original** SWAN-SF multi-class labels and set
`training.task: multiclass` in `config.yaml` — the same pipeline handles it.

---

## API

| Endpoint | Returns |
|---|---|
| `GET /` | the forecast web page |
| `GET /health` | liveness + whether the SHARP model is loaded |
| `GET /api/forecast` | full prediction: nowcast + 12/24/48 h + ensemble + tracks |
| `GET /api/flux` | recent GOES X-ray series (for the chart) |
| `GET /api/regions` | current active regions + NOAA per-region probabilities |

---

## Resilience / failsafes

Every live fetch (`solarflare/sources.py`):

1. tries multiple URLs in order (primary → secondary → shorter feed),
2. has a hard timeout,
3. **caches** each success to `.cache/` (the failsafe backup),
4. falls back to the most recent cache if all sources fail, clearly flagged
   "cached / N min old",
5. if *everything* is down, the predictor returns a **climatological fallback**
   forecast plus a notice — it never errors out.

Live sources used: NOAA SWPC GOES X-ray (primary + secondary + 6-hour),
NOAA Solar Region Summary, NOAA GOES flare events, and NASA DONKI (optional
key). Add your free key at <https://api.nasa.gov> in `config.yaml`.

---

## Make it better (it's a *first* model on purpose)

* Train on the real SWAN-SF partitions (`download` → `train`).
* Add multi-class labels for true upper-C / M / X separation.
* Wire **live SHARP** from JSOC/SDO so the ML track runs in production
  (`predictor.sharp_forecast` already accepts a feature vector).
* Swap `HistGradientBoosting` for LightGBM/XGBoost or an LSTM in
  `solarflare/train.py` — the data + eval harness stay the same.
* Tune the flux-persistence priors in `solarflare/fluxmodel.py` against the
  GOES flare catalogue.

---

## Honest limitations

* Solar-flare forecasting has a **hard skill ceiling**; no model gets near
  "certainty". Treat outputs as probabilistic guidance alongside official
  [NOAA SWPC](https://www.swpc.noaa.gov/) products.
* The always-on web forecast runs the **flux + NOAA** tracks; the **SHARP ML**
  track activates only when magnetic features are supplied (the synthetic smoke
  model proves the plumbing).
* `DEMO_KEY` for NASA DONKI is heavily rate-limited — add your own key.

---

## File map

```
config.yaml                 all thresholds, paths, sources, training options
solarflare/
  config.py    labels.py    config loader · flare taxonomy + X-warnings
  data.py      train.py     SWAN-SF loader + features · trainer (.pkl/.joblib)
  evaluate.py  download.py  TSS/HSS skill scores · dataset downloader
  sources.py   nowcast.py   resilient live fetchers · current-state from flux
  fluxmodel.py predictor.py persistence forecaster · orchestrator + fallbacks
api/server.py                FastAPI backend + serves the frontend
frontend/index.html          the forecast page (dark ops dashboard)
scripts/smoke_test.py        end-to-end proof with no download
```
