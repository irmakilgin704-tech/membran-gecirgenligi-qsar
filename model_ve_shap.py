import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Hazir veri setini yukle
print("Veri seti yukleniyor...")
try:
    df = pd.read_csv("hazir_veri_seti.csv")
except FileNotFoundError:
    print("Hata: 'hazir_veri_seti.csv' bulunamadi! Lutfen once veri hazirlama kodunu calistirin.")
    exit()

print(f"Toplam molekul sayisi: {len(df)}")
print("\nMembran Dagilimi:")
print(df['membrane_type'].value_counts())

# 2. Ozellikler (X) ve Hedef Degisken (y) Secimi
feature_cols = ['MolLogP', 'MW', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'HeavyAtoms']

X = df[feature_cols]
y = df['permeability_value'] # Gecirgenlik degeri

# 3. Egitim ve Test setlerine bolme (Kucuk set oldugu icin test oranini dengeliyoruz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. XGBoost Modelini Egitme
print("\nXGBoost modeli egitiliyor...")
model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Model basari testi
y_pred = model.predict(X_test)
print(f"Model R2 Skoru: {r2_score(y_test, y_pred):.3f}")
print(f"Hata (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# 5. SHAP Analizi (Aciklanabilir Yapay Zeka)
print("\nSHAP aciklamalari hesaplaniyor...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X)

# 6. SHAP Ozet Grafigini Cizme ve Kaydetme
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("TÜBİTAK 2204-A: Gecirgenligi Etkileyen Faktorler (SHAP)", fontsize=12)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)
print("\nSHAP ozet grafigi 'shap_summary_plot.png' olarak basariyla kaydedildi!")
plt.show()