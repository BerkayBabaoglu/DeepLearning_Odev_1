import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from logisticregression import X_test_scaled, X_train_scaled

#yine pseudo 
#veri yukleme
#degiskenleri ayirma
#egitim-test-dogrulama olaylari
#olceklendirme
#model egitimi (random forest)
#tahmin
#degerlendirme
#bitti

veri = pd.read_csv("data.csv")
veri = veri.drop(columns=["id"])
veri = veri.dropna(axis=1, how="all")
veri["diagnosis"] = veri["diagnosis"].map({"M":1,"B":0})

print("veri setinin boyutu", veri.shape)
print("sinif dagilimi: iyi huylu:", sum(veri["diagnosis"] == 0), "kotu huylu:", sum(veri["diagnosis"] == 1))

#egitim 70, test-dogrulama 15 15 yine ayni

X = veri.drop(columns=["diagnosis"])
Y = veri["diagnosis"]

X_train, X_temp, Y_train, Y_temp = train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y)

X_val,X_test, Y_val,Y_test = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=42,stratify=Y_temp)

print(f"egitim seti: {X_train.shape[0]} ornek(%70)")
print(f"dogrulama seti: {X_val.shape[0]} ornek(%15)")
print(f"test seti: {X_test.shape[0]} ornek(%15)")

#olcekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#random forest icin 100 agac 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, Y_train)

Y_val_pred = model.predict(X_val_scaled)
print("dogrulama seti sonuclari")
print(f"Accuracy:  {accuracy_score(Y_val,Y_val_pred):.4f}")
print(f"Precision: {precision_score(Y_val, Y_val_pred):.4f}")
print(f"Recall:    {recall_score(Y_val, Y_val_pred):.4f}")
print(f"F1-Score:  {f1_score(Y_val, Y_val_pred):.4f}")

Y_test_pred = model.predict(X_test_scaled)
print("test seti sonuclari")
print(f"Accuracy:  {accuracy_score(Y_test, Y_test_pred):.4f}")
print(f"Precision: {precision_score(Y_test, Y_test_pred):.4f}")
print(f"Recall:    {recall_score(Y_test, Y_test_pred):.4f}")
print(f"F1-Score:  {f1_score(Y_test, Y_test_pred):.4f}")

print("karisiklik matrisi:")
print(confusion_matrix(Y_test, Y_test_pred))

print("siniflandirma raporu:(randomforest)")
print(classification_report(Y_test, Y_test_pred, target_names=["Benign", "Malignant"]))