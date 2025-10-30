import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score
import joblib


df = pd.read_csv("heart_disease_uci (1).csv")


df = df.drop(columns=["id","dataset","ca","thal","slope"])
df = df.dropna(subset=["fbs","exang","restecg","trestbps","chol","oldpeak"])
df = pd.get_dummies(df,columns=["sex","fbs","exang"],drop_first=True,dtype=int)
df[["thalch", "chol", "trestbps","oldpeak"]] = df[["thalch", "chol", "trestbps","oldpeak"]].astype(int)

le=LabelEncoder()
df["cp"]= le.fit_transform(df["cp"])

le2=LabelEncoder()
df["restecg"]= le2.fit_transform(df["restecg"])



x = df.drop(columns=["num"])
y = df["num"].apply(lambda i: 0 if i==0 else 1)



x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)




st.title("Kalp Hastalığı Tahmin Uygulaması ❤️")


mapping_cp = {
    "Tipik Anjina": 0,
    "Atipik Anjina": 1,
    "Anjinal Olmayan": 2,
    "Asemptomatik": 3
}

mapping_sex={
    "Erkek": 1,
    "Kadın": 0
}

mapping_fbs={
    "Evet": 1,
    "Hayır": 0
}

mapping_exang={
    "Evet": 1,
    "Hayır": 0
}

mapping_res = {
    "Normal": 0,
    "ST-T Dalga Anormalliği": 1,
    "Sol Ventrikül Hipertrofisi": 2
}



# Kullanıcıdan input al
st.sidebar.header("Girdi Bilgileri")
age = st.sidebar.number_input("Yaş",10,100, int(df["age"].mean()))
thalch = st.sidebar.number_input("Maks. Kalp Hızı (thalch)", int(df["thalch"].min()), int(df["thalch"].max()), int(df["thalch"].mean()))
trestbps = st.sidebar.number_input("Dinlenme Halindeki Tansiyon(trestbps)", int(df["trestbps"].min()), int(df["trestbps"].max()), int(df["trestbps"].mean()))
chol = st.sidebar.number_input("Kolesterol (chol)", int(df["chol"].min()), int(df["chol"].max()), int(df["chol"].mean()))
oldpeak = st.sidebar.number_input("ST Depresyon,EKG Sonucu (oldpeak)", int(df["oldpeak"].min()), int(df["oldpeak"].max()), int(df["oldpeak"].mean()))

# Diğer binary özellikler örnek olarak 0-1
sex_choice = st.sidebar.selectbox("Cinsiyet ",  options=list(mapping_sex.keys()))
sex=mapping_sex[sex_choice]

cp_choice = st.sidebar.selectbox("Göğüs ağrısı tipi ", options=list(mapping_cp.keys()))
cp= mapping_cp[cp_choice]

fbs_choice= st.sidebar.selectbox("Kan şekeri >120 mg/dl üstü mü?", options=list(mapping_fbs.keys()))
fbs= mapping_fbs[fbs_choice]

restecg_choice = st.sidebar.selectbox("EKG sonucu", options=list(mapping_res.keys()))
restecg= mapping_res[restecg_choice]


exang_choice= st.sidebar.selectbox("Egzersize bağlı göğüs ağrısı var mı?",options=list(mapping_exang.keys()))
exang= mapping_exang[exang_choice]




# Kullanıcı girdi verisi
input_data = np.array([[age, thalch, trestbps, chol, oldpeak, sex, cp, fbs, restecg, exang]])

# Tahmin butonu
if st.button("Tahmin Et"):
    y_prob = model.predict_proba(input_data)[:,1]
    # ROC optimizasyonu gibi işlemler
    fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    st.success(f"Kalp hastalığı riski tahmini: {'Var' if y_pred[0]==1 else 'Yok'}")

