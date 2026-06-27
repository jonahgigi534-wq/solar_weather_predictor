"""FastAPI backend serving the solar-flare forecast.

Run:
    uvicorn api.server:app --reload --port 8000
    # or: python -m api.server

Endpoints
    GET /health            -> liveness + whether the SHARP model is loaded
    GET /api/forecast      -> the full prediction (nowcast + 12/24/48h forecast)
    GET /api/flux          -> raw recent GOES X-ray series (for the chart)
    GET /                  -> serves the forecast web page (frontend/index.html)
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from solarflare import predictor, sources
from solarflare.config import load_config

cfg = load_config()
app = FastAPI(title="Solar Flare Predictor", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONTEND = os.path.join(cfg["_project_root"], "frontend", "index.html")


@app.get("/health")
def health():
    model = predictor.load_sharp_model(cfg)
    return {
        "status": "ok",
        "sharp_model_loaded": model is not None,
        "sharp_test_tss": (model or {}).get("metrics", {}).get("tss") if model else None,
    }


@app.get("/api/forecast")
def forecast():
    # SHARP features are not supplied in the always-on web path (live SHARP
    # needs JSOC); the flux track + ensemble still produce a full forecast.
    return JSONResponse(predictor.predict(cfg=cfg))


@app.get("/api/flux")
def flux():
    res = sources.get_xray_flux(cfg)
    if not res.ok:
        return JSONResponse({"ok": False, "error": res.error}, status_code=503)
    # Downsample to keep the payload light for the chart (~every 5 min).
    rows = res.data[::5]
    return {
        "ok": True,
        "status": res.status,
        "series": [{"t": r["time_tag"], "flux": r["flux"]} for r in rows if r.get("flux")],
    }


@app.get("/api/regions")
def regions():
    res = sources.get_solar_regions(cfg)
    if not res.ok or not isinstance(res.data, list):
        return JSONResponse({"ok": False, "regions": []}, status_code=503)
    keep = ("region", "spot_class", "mag_class", "number_spots", "area",
            "c_flare_probability", "m_flare_probability", "x_flare_probability")
    rows = [{k: r.get(k) for k in keep} for r in res.data]
    return {
        "ok": True,
        "status": res.status,
        "observed_date": res.data[0].get("observed_date") if res.data else None,
        "regions": rows,
    }


@app.get("/")
def index():
    if os.path.exists(FRONTEND):
        # No-cache so a redeployed page is always picked up by the browser.
        return FileResponse(FRONTEND, headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)


def main():
    import uvicorn
    uvicorn.run("api.server:app", host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()
