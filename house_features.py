"""
house_features.py
=================
الملف الوحيد المسؤول عن تجهيز بيانات البيت (Feature Engineering + Target Encoding).

نفس هذا الكود يُستخدم في:
  1) تدريب النموذج داخل Jupyter
  2) التنبؤ لاحقًا داخل Flask
وبذلك يتحقق الشرط:  Training preprocessing = Prediction preprocessing

مهم: ضع هذا الملف بجانب الـ Notebook، وانسخه لاحقًا إلى مجلد backend (أو اجعله مشتركًا)
لأن joblib يحتاج استيراد نفس الكلاس عند تحميل النموذج.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

# ------------------------------------------------------------------ ثوابت
SEATTLE_LAT, SEATTLE_LONG = 47.6062, -122.3321
REFERENCE_YEAR = 2015          # آخر سنة بيع في البيانات (2014-2015): نحسب العمر بالنسبة لها
SMOOTHING = 10                 # قوة تنعيم Target Encoding للمناطق قليلة البيوت

# الحقول التي سيدخلها المستخدم في الواجهة (هذه فقط متاحة وقت التنبؤ)
USER_INPUT_COLUMNS = [
    "zipcode", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_basement",
    "yr_built", "yr_renovated",
]

# الميزات النهائية التي يتدرب عليها النموذج (بهذا الترتيب بالضبط)
#ملاحظة المستخدم في اللي قبل اليوسر ان بوت كان هناك 13 عمود فقط اما هنا 17 اي 7ميزات مشتقة من ال13
FEATURE_COLUMNS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_basement",
    "house_age", "was_renovated", "basement_ratio", "distance_from_seattle",
    "bath_bed_ratio", "relative_size_to_neighbors", "location_score",
]


def _smoothed_zip_mean(zips, y, global_mean, m=SMOOTHING):
    """متوسط سعر كل منطقة بعد تنعيمه نحو المتوسط العام (يحمي المناطق ذات العدد القليل)."""
    g = pd.DataFrame({"z": np.asarray(zips), "y": np.asarray(y, dtype=float)}).groupby("z")["y"].agg(["sum", "count"])
    return (g["sum"] + m * global_mean) / (g["count"] + m)


class HouseFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    fit(X, y)  : يتعلم من بيانات التدريب فقط (متوسط سعر المنطقة، مركز المنطقة، متوسط حجم البيوت فيها)
    transform  : يحوّل بيانات المستخدم الخام إلى FEATURE_COLUMNS
    fit_transform : يستخدم Out-of-Fold لـ location_score حتى لا يرى صف التدريب سعره هو

    عند fit تحتاج الأعمدة: zipcode, lat, long, sqft_living (+ y)
    عند transform تحتاج فقط USER_INPUT_COLUMNS.
    """

    def __init__(self, smoothing=SMOOTHING, n_splits=5, random_state=42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    # ---------------------------------------------------------------- learn
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        self.global_mean_price_ = float(y.mean())
        self.zip_price_map_ = _smoothed_zip_mean(X["zipcode"], y, self.global_mean_price_, self.smoothing)
        self.zip_lat_ = X.groupby("zipcode")["lat"].mean()
        self.zip_long_ = X.groupby("zipcode")["long"].mean()
        self.zip_living_median_ = X.groupby("zipcode")["sqft_living"].median()
        self.global_lat_ = float(X["lat"].mean())
        self.global_long_ = float(X["long"].mean())
        self.global_living_median_ = float(X["sqft_living"].median())
        self.known_zipcodes_ = sorted(self.zip_price_map_.index.tolist())
        return self
        #يتعلم من بيانات التدريب فقط

    # ------------------------------------------------------------ build
    def _build(self, X, location_score):
        X = X.reset_index(drop=True)
        z = X["zipcode"].astype(int)
        lat = z.map(self.zip_lat_).fillna(self.global_lat_)
        lon = z.map(self.zip_long_).fillna(self.global_long_)
        zliv = z.map(self.zip_living_median_).fillna(self.global_living_median_)

        out = pd.DataFrame(index=X.index)
        #الاعمدة الاصليه تبع المشروع
        for c in ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                  "waterfront", "view", "condition", "grade", "sqft_basement"]:
            out[c] = X[c].astype(float)
            #الميزات المشتقة بعنى التي تم اشتقاقها من الاصلية
        out["house_age"] = (REFERENCE_YEAR - X["yr_built"]).clip(lower=0).astype(float)
        out["was_renovated"] = (X["yr_renovated"] > 0).astype(float)
        out["basement_ratio"] = X["sqft_basement"] / X["sqft_living"]
        out["distance_from_seattle"] = np.sqrt((lat - SEATTLE_LAT) ** 2 + (lon - SEATTLE_LONG) ** 2) * 111
        out["bath_bed_ratio"] = X["bathrooms"] / X["bedrooms"].replace(0, 1)
        out["relative_size_to_neighbors"] = X["sqft_living"] / zliv
        out["location_score"] = np.asarray(location_score, dtype=float)
        return out[FEATURE_COLUMNS]

    # -------------------------------------------------------- inference
    def transform(self, X):
        ls = X["zipcode"].astype(int).map(self.zip_price_map_).fillna(self.global_mean_price_)
        return self._build(X, ls)

    # --------------------------------------------------------- training
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        oof = np.full(len(X), np.nan)
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr_idx, va_idx in kf.split(X):
            gm = y.iloc[tr_idx].mean()
            m = _smoothed_zip_mean(X["zipcode"].iloc[tr_idx], y.iloc[tr_idx], gm, self.smoothing)
            oof[va_idx] = X["zipcode"].iloc[va_idx].map(m).fillna(gm).values
        return self._build(X, oof)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLUMNS)


def to_input_frame(houses):
    """يحوّل dict (أو قائمة dicts) من المستخدم إلى DataFrame بالأعمدة المطلوبة فقط."""
    if isinstance(houses, dict):
        houses = [houses]
    df = pd.DataFrame(houses)
    missing = [c for c in USER_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"حقول ناقصة: {missing}")
    return df[USER_INPUT_COLUMNS]
    #وظيفتة تحويل list او dict  الى Dataframe
    #تتاكد مافيه حقول ناقصة
    #ترتيب الاعمدة ب الترتيب الصحيح
    
