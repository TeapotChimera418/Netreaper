import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# 1. COLUMN NAMES
# =========================

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv(r"F:\NetReaper\KDDTrain.csv", names=columns)
df.to_csv(r"F:\NetReaper\KDDTrain_with_headers.csv", index=False)

# =========================
# 3. DROP UNUSED
# =========================

df.drop("difficulty", axis=1, inplace=True)

# =========================
# 4. LABEL → BINARY
# =========================

df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# =========================
# 5. ENCODE CATEGORICAL
# =========================

categorical_cols = ["protocol_type", "service", "flag"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =========================
# 6. OPTIONAL (FASTER)
# =========================

df = df.sample(20000, random_state=42)

# =========================
# 7. SPLIT
# =========================

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 8. SCALE
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 9. CHECK
# =========================

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y distribution:\n", y.value_counts())