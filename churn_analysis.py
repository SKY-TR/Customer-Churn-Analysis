import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Veri Simülasyonu (Banka Müşteri Hareketleri)
data = {
    'tenure': np.random.randint(1, 72, 1000), # Kaç aydır müşteri?
    'monthly_charges': np.random.randint(20, 150, 1000),
    'total_charges': np.random.randint(100, 5000, 1000),
    'churn': np.random.randint(0, 2, 1000) # 0: Kaldı, 1: Gitti
}
df = pd.DataFrame(data)

# 2. Model Hazırlığı
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Gelişmiş Algoritma (Gradient Boosting)
# Öğrenci seviyesinde "ben farklı algoritmalar da biliyorum" demek için iyi bir seçim.
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# 4. Sonuç
score = accuracy_score(y_test, model.predict(X_test))
print(f"Müşteri Terk Tahminleme Başarısı: %{score*100:.2f}")
