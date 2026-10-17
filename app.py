"""SMART HOUSE AI — Flask backend. التشغيل:  python app.py  ثم افتح http://127.0.0.1:5000"""
import json, os, sys, logging
import numpy as np, pandas as pd, joblib
from flask import Flask, Response, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from house_features import to_input_frame, FEATURE_COLUMNS, USER_INPUT_COLUMNS

ART = os.path.join(BASE, "artifacts")
META = json.load(open(os.path.join(ART, "model_metadata.json"), encoding="utf-8"))
PIPE = joblib.load(os.path.join(ART, META["pipeline_file"]))      # لا إعادة تدريب: نحمّل المحفوظ فقط
FB, MDL, PRE = PIPE.named_steps["features"], PIPE.named_steps["model"], PIPE[:-1]
ZIPS = {int(z): {"lat": float(FB.zip_lat_[z]), "lng": float(FB.zip_long_[z]),
                 "score": float(FB.zip_price_map_[z])} for z in FB.known_zipcodes_}
LABELS = {"bedrooms": "غرف النوم", "bathrooms": "الحمامات", "sqft_living": "مساحة المعيشة", "sqft_lot": "مساحة الأرض",
          "floors": "عدد الأدوار", "waterfront": "الإطلالة المائية", "view": "جودة الإطلالة", "condition": "حالة العقار",
          "grade": "جودة البناء", "sqft_basement": "مساحة القبو", "house_age": "عمر العقار", "was_renovated": "الترميم",
          "basement_ratio": "نسبة القبو", "distance_from_seattle": "البعد عن وسط سياتل",
          "bath_bed_ratio": "نسبة الحمامات للغرف", "relative_size_to_neighbors": "الحجم مقارنة بالمنطقة",
          "location_score": "قيمة المنطقة السوقية"}
def _num(x, default):
    try: return float(x)
    except (TypeError, ValueError): return default
RANGES = {k: {"min": _num(v.get("min"), 0.0), "max": _num(v.get("max"), 1e9)}      # نضمن أن الحدود أرقام دائمًا
          for k, v in META["input_ranges"].items()}
TRAIN_RANGES = {k: dict(v) for k, v in RANGES.items()}          # حدود بيانات التدريب الفعلية (للتنبيه فقط)
RANGES["bathrooms"]["min"], RANGES["bathrooms"]["max"] = 1, 6                                    # حدود الواجهة المطلوبة
RANGES["floors"]["max"] = 3     # أقصى عدد أدوار موثوق في بيانات التدريب (الحد الفعلي 3.5)
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def compute_shap(Xm):
    """SHAP الحقيقي. لنموذج XGBoost نستخدم TreeSHAP المدمج فيه (نفس القيم، وأكثر ثباتًا مع النسخ الحديثة)."""
    if type(MDL).__name__ == "XGBRegressor":
        try:
            import xgboost as xgb
            c = MDL.get_booster().predict(xgb.DMatrix(Xm), pred_contribs=True)
            return np.asarray(c[:, :-1]), float(c[0, -1])
        except Exception:
            app.logger.exception("xgboost pred_contribs failed, trying shap library")
    import shap
    ex = shap.TreeExplainer(MDL)
    return np.asarray(ex.shap_values(Xm)), float(np.ravel(ex.expected_value)[0])

def validate(d):
    if not isinstance(d, dict): raise ValueError("بيانات غير صالحة")
    out = {}
    for k in USER_INPUT_COLUMNS:
        if k not in d or d[k] in ("", None): raise ValueError(f"الحقل مطلوب: {k}")
        try: v = float(d[k])
        except (TypeError, ValueError): raise ValueError(f"قيمة غير صالحة: {k}")
        if k == "zipcode":
            if int(v) not in ZIPS: raise ValueError("الرمز البريدي خارج نطاق بيانات مقاطعة King")
            v = int(v)
        elif k == "yr_renovated":
            if v != 0 and not (RANGES[k]["min"] <= v <= 2015): raise ValueError("سنة الترميم غير منطقية")
        elif k in RANGES and not (RANGES[k]["min"] <= v <= RANGES[k]["max"]):
            raise ValueError(f"{LABELS.get(k, k)} خارج النطاق ({RANGES[k]['min']} – {RANGES[k]['max']})")
        out[k] = v
    if out["sqft_basement"] > out["sqft_living"]: raise ValueError("القبو أكبر من مساحة المعيشة")
    if out["yr_renovated"] and out["yr_renovated"] < out["yr_built"]: raise ValueError("الترميم قبل سنة البناء")
    return out

def fmt(f, v):
    if f in ("waterfront", "was_renovated"): return "نعم" if v >= .5 else "لا"
    if f == "distance_from_seattle": return f"{v:.1f} كم"
    if f == "house_age": return f"{v:.0f} سنة"
    if f == "location_score": return f"${v:,.0f}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"

@app.route("/")
def index(): return render_template("index.html")

# صفحة الخريطة منفصلة (تُعرض داخل iframe) حتى لا تتأثر بتصميم الصفحة العربية. تعمل مثل خريطة Leaflet العادية.
MAP_HTML = """<!DOCTYPE html><html lang="en" dir="ltr"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>html,body,#m{height:100%;margin:0}</style></head><body><div id="m"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js" onerror="parent.postMessage({type:'nomap'},'*')"></script>
<script>
if (typeof L === 'undefined') { parent.postMessage({type:'nomap'}, '*'); }
else fetch('/api/meta').then(r => r.json()).then(M => {
  const z = Object.values(M.zips), la = z.map(p => p.lat), ln = z.map(p => p.lng);
  const bb = [[Math.min(...la) - .06, Math.min(...ln) - .06], [Math.max(...la) + .06, Math.max(...ln) + .06]];
  const map = L.map('m').fitBounds(bb);
  const P = [
    ['https://tile.openstreetmap.org/{z}/{x}/{y}.png', '&copy; OpenStreetMap contributors'],
    ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', 'Tiles &copy; Esri'],
    ['https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', '&copy; OpenStreetMap France'],
    ['https://tile.openstreetmap.de/{z}/{x}/{y}.png', '&copy; OpenStreetMap']];
  let pi = 0, ok = 0, bad = 0;
  const tl = L.tileLayer(P[0][0], {maxZoom: 18}).addTo(map);
  map.attributionControl.addAttribution(P[0][1]);
  const next = () => {
    if (pi >= P.length - 1) { parent.postMessage({type: 'nomap'}, '*'); return; }
    map.attributionControl.removeAttribution(P[pi][1]); pi++; ok = 0; bad = 0;
    tl.setUrl(P[pi][0]); map.attributionControl.addAttribution(P[pi][1]);
  };
  tl.on('tileload', () => { ok++; });
  tl.on('tileerror', () => { if (ok === 0 && ++bad >= 3) next(); });
  setInterval(() => { if (ok === 0) next(); }, 7000);
  L.rectangle(bb, {color: '#d4a017', weight: 2, dashArray: '7 7', fill: false, interactive: false}).addTo(map);
  const pin = '<svg width="30" height="42" viewBox="0 0 30 42"><path d="M15 1C7 1 1 7 1 15c0 10 14 26 14 26s14-16 14-26C29 7 23 1 15 1z" fill="#e8504a" stroke="#fff" stroke-width="2"/><circle cx="15" cy="15" r="5.5" fill="#fff"/></svg>';
  const mk = L.marker([47.6, -122.2], {icon: L.divIcon({className: '', html: pin, iconSize: [30, 42], iconAnchor: [15, 41]})}).addTo(map);
  map.on('click', e => parent.postMessage({type: 'pick', lat: e.latlng.lat, lng: e.latlng.lng}, '*'));
  addEventListener('message', e => { const d = e.data || {};
    if (d.type === 'setZip') { mk.setLatLng([d.lat, d.lng]); d.fly ? map.flyTo([d.lat, d.lng], 12) : map.panTo([d.lat, d.lng]); }
    if (d.type === 'resize') map.invalidateSize(); });
  parent.postMessage({type: 'mapready'}, '*');
});
</script></body></html>"""

@app.route("/map")
def map_page():
    return Response(MAP_HTML, mimetype="text/html")

_PTS = None
@app.route("/api/points")
def points():
    """نقاط حقيقية من بياناتك (lat, long, السعر بالآلاف) لرسم خريطة الأسعار بدون أي إنترنت."""
    global _PTS
    if _PTS is None:
        d = pd.read_csv(os.path.join(BASE, "kc_house_data.csv"), usecols=["lat", "long", "price"])
        _PTS = [[round(a, 4), round(b, 4), int(p / 1000)] for a, b, p in zip(d["lat"], d["long"], d["price"])]
    return jsonify({"pts": _PTS})

@app.route("/api/meta")
def meta():
    keep = ("model_name", "model_class", "model_version", "metrics_test", "cv_r2", "model_comparison_test",
            "model_selection_cv_r2", "feature_count", "dataset", "input_ranges", "reference_year")
    m = {k: META.get(k) for k in keep}
    m.update(zips=ZIPS, labels=LABELS, input_ranges=RANGES, global_importance=(META.get("shap") or {}).get("global_importance_mean_abs"))
    return jsonify(m)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        house = validate(request.get_json(silent=True))
        frame = to_input_frame(house)
        price = float(PIPE.predict(frame)[0])                  # نفس تجهيز التدريب 100%
        Xm = PRE.transform(frame)
        Xm = Xm if isinstance(Xm, pd.DataFrame) else pd.DataFrame(Xm, columns=FEATURE_COLUMNS)
        try:
            sv, base = compute_shap(Xm)
            row = Xm.iloc[0]
            items = sorted(({"feature": f, "label": LABELS[f], "display": fmt(f, row[f]), "shap": float(v)}
                            for f, v in zip(FEATURE_COLUMNS, sv[0])), key=lambda x: -abs(x["shap"]))
        except Exception:
            app.logger.exception("SHAP failed (prediction is still returned)")
            base, items = price, []
        sent = lambda i: f"{i['label']} ({i['display']}) " + ("رفع" if i["shap"] > 0 else "خفّض") + \
                         f" القيمة المتوقعة بنحو ${abs(i['shap']):,.0f}"
        pos = [i for i in items if i["shap"] > 0][:3]; neg = [i for i in items if i["shap"] < 0][:3]
        warns = [f"{LABELS.get(k, k)} ({house[k]:g}) خارج نطاق بيانات التدريب" for k in TRAIN_RANGES
                 if k in house and k not in ("yr_renovated", "waterfront") and
                 not (TRAIN_RANGES[k]["min"] <= house[k] <= TRAIN_RANGES[k]["max"])]
        return jsonify({"warnings": warns, "predicted_price": price, "base_value": base, "contributions": items,
                        "top_positive": pos, "top_negative": neg,
                        "explanations": {"positive": [sent(i) for i in pos], "negative": [sent(i) for i in neg]},
                        "mae": META["metrics_test"]["MAE"], "model": META["model_name"],
                        "r2": META["metrics_test"]["R2"], "zip": house["zipcode"]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        app.logger.exception("prediction failed")
        return jsonify({"error": "تعذّر إتمام التحليل الآن. حاول مرة أخرى.",
                        "detail": f"{type(e).__name__}: {str(e)[:300]}"}), 500   # detail للتشخيص فقط، احذفه قبل التسليم

@app.route("/api/selftest")
def selftest():
    """فحص تشخيصي: افتح 127.0.0.1:5000/api/selftest لمعرفة أي مرحلة تفشل."""
    out = {"model": META["model_name"]}
    house = {"zipcode": 98052, "bedrooms": 3, "bathrooms": 2, "sqft_living": 2200, "sqft_lot": 5000, "floors": 1,
             "waterfront": 0, "view": 0, "condition": 3, "grade": 8, "sqft_basement": 0, "yr_built": 2005, "yr_renovated": 0}
    try:
        frame = to_input_frame(house); out["prediction"] = float(PIPE.predict(frame)[0])
    except Exception as e:
        out["prediction_error"] = repr(e); return jsonify(out)
    try:
        Xm = PRE.transform(frame); sv, base = compute_shap(Xm); out["shap"] = "ok"; out["shap_values"] = int(sv.shape[1])
    except Exception as e:
        out["shap_error"] = repr(e)
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
