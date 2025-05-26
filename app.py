
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="VP-AI", layout="wide")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

def convert_binary(val):
    if isinstance(val, str):
        val = val.strip().lower()
        if val in ["x", "có", "yes"]:
            return 1
        elif val in ["/", "khong", "không", "no", "ko"]:
            return 0
    return val

def convert_age(val):
    if isinstance(val, str):
        val = val.strip()
        if "thg" in val.lower():
            return float(val.replace("Thg", "").replace("thg", "")) / 10
        try:
            return float(val)
        except:
            return np.nan
    return val

def convert_numeric(val):
    try:
        return float(str(val).strip())
    except:
        return np.nan

@st.cache_data
def load_model():
    df = pd.read_csv("Mô hình AI.csv")
    df.columns = df.columns.str.strip()
    df.drop(columns=["So ngay dieu tri"], errors="ignore", inplace=True)
    if "Benh ngay thu truoc khi nhap vien" in df.columns:
        df.rename(columns={"Benh ngay thu truoc khi nhap vien": "Benh ngay thu"}, inplace=True)
    if " SpO2" in df.columns:
        df.rename(columns={"SpO2": "SpO2"}, inplace=True)

    df["Tuoi"] = df["Tuoi"].apply(convert_age)
    df["Benh ngay thu"] = df["Benh ngay thu"].apply(convert_numeric)
    df["SpO2"] = df["SpO2"].apply(convert_numeric)

    binary_cols = df.select_dtypes(include="object").columns.difference(["Tac nhan", "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"])
    df[binary_cols] = df[binary_cols].applymap(convert_binary)

    df = df[df["Tac nhan"].notna()]
    X = df.drop(columns=["Tac nhan", "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"], errors="ignore")
    y = df["Tac nhan"]

    # Loại các cột kháng sinh theo danh sách cụ thể
    abx_keywords = ["amoxicilin", "ceftriaxone", "vancomycin", "meropenem", "levofloxacin", "clarithromycin", "penicillin", "clindamycin"]
    cols_to_remove = ['Benzylpenicillin', 'Ceftriaxone', 'Levofloxacin', 'Moxifloxacin', 'Erythomycin', 'Clindamycin', 'Linezolid', 'Cefotaxime', 'Vancomycin', 'Tetracyline', 'Tigecycline', 'Chloranphenicol', 'Rifampicin', 'Trimethoprim', 'Fusidic acid', 'Oxacillin', 'Gentamicin', 'Ciprofloxacin', 'Teicoplanin', 'Meropenem', 'Arithromycin', 'Ampicillin', 'Ampicillin-Sulbalactam', 'Piperacillin', 'Piperacillin/ Tazobactam', 'Cefuroxime', 'Cefuroxime Axetil', 'Ceftazidine', 'Ertapenem', 'Imipenem', 'Viprofloxacin', 'Amoxicilin clavulanic']
    X = X.drop(columns=cols_to_remove, errors="ignore")

    X = X.applymap(lambda x: x if isinstance(x, (int, float)) or pd.isnull(x) else np.nan)
    X = X.fillna(0)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_cols = load_model()

st.markdown("### 📋 Nhập dữ liệu lâm sàng")

user_input = {}
for idx, col in enumerate(feature_cols):
    safe_key = f"input_{idx}"
    if col == "Tuoi":
        user_input[col] = st.number_input("Tuổi (năm)", min_value=0.0, max_value=120.0, step=1.0, key=safe_key)
    elif col in ["Nhiet do", "Bach cau", "CRP", "Nhip tho", "Mach", "Benh ngay thu", "SpO2"]:
        user_input[col] = st.number_input(col, value=0.0, key=safe_key)
    else:
        user_input[col] = st.radio(f"**Amoxicilin clavulanic**:", ["Không", "Có"], horizontal=True, key=safe_key) == "Có"

if st.button("🔍 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if isinstance(input_df[col].iloc[0], bool):
            input_df[col] = input_df[col].astype(int)

    pred = model.predict(input_df)[0]
    st.success(f"✅ Tác nhân gây bệnh được dự đoán: **{pred}**")

    khang_sinh = {'H. influenzae': ['Amoxicilin clavulanic', 'Ceftriaxone'], 'K. pneumonia': ['Meropenem', 'Ceftriaxone'], 'M. catarrhalis': ['Amoxicilin clavulanic', 'Clarithromycin'], 'M. pneumonia': ['Clarithromycin', 'Levofloxacin'], 'RSV': [], 'S. aureus': ['Vancomycin', 'Clindamycin'], 'S. epidermidis': ['Vancomycin'], 'S. mitis': ['Penicillin'], 'S. pneumonia': ['Ceftriaxone', 'Vancomycin'], 'unspecified': []}
    abx_list = khang_sinh.get(pred, [])
    st.markdown("### 💊 Kháng sinh gợi ý:")
    if abx_list:
        for abx in abx_list:
            st.write(f"- **{abx}**")
    else:
        st.info("Không có kháng sinh nào được gợi ý.")
