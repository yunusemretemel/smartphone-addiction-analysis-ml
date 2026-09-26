import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. VERİ YÜKLEME
dosya_adi = r'C:\Users\yunus\OneDrive\Masaüstü\Akıllı Telefon Bağımlılığı Analizi\veri.csv'
df = pd.read_csv(dosya_adi)
hedef_sutun = 'addiction_level' 

# 2. X (Özellikler) ve y (Hedef) OLARAK AYIRMA
X = df.drop([hedef_sutun, 'user_id'], axis=1)
y = df[hedef_sutun]

# KELİMELERİ SAYIYA ÇEVİRME
X = pd.get_dummies(X, drop_first=True)
y = LabelEncoder().fit_transform(y)

# 3. VERİYİ BÖLME
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ÖLÇEKLENDİRME (SCALING)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. MODELLERİ EĞİTME
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 6. TAHMİN VE SONUÇLAR
knn_pred = knn_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

print("--- SONUÇLAR ---")
print(f"KNN Modeli Doğruluk Oranı: % {accuracy_score(y_test, knn_pred) * 100:.2f}")
print(f"SVM Modeli Doğruluk Oranı: % {accuracy_score(y_test, svm_pred) * 100:.2f}")

# 7. GRAFİKLER
# Pencere genişliğini 12'den 15'e çıkararak grafiklere yer açtık
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# KNN - Confusion Matrix (Klasik Mavi)
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('KNN - Confusion Matrix', pad=15)
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gerçek Değer')

# SVM - Confusion Matrix (Klasik Yeşil)
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SVM - Confusion Matrix', pad=15)
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gerçek Değer')

# İki grafik arasındaki yatay boşluğu (w_pad) 6.0 yaparak mesafeyi belirgin şekilde açtık
plt.tight_layout(w_pad=6.0)
plt.show()