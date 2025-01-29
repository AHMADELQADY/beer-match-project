import logging
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 📂 Creazione della cartella per i log
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 🔍 Configurazione del logger per il training
logging.basicConfig(
    filename=os.path.join(log_dir, "training_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📥 1. Caricare il dataset
df = pd.read_csv("cibo_birra_dataset_completo.csv")

# 🛠 2. Preprocessing: Convertire variabili categoriche in numeri
categorical_columns = ["Colore", "Gusto", "Corpo/consistenza", "Carbonatazione", "Origine"]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Salviamo i LabelEncoders per l'inferenza

# 🔥 3. Salvare gli encoder per usarli durante l’inferenza
joblib.dump(label_encoders, "label_encoders.pkl")

# 🔄 4. Separazione delle feature e del target
X = df.drop(columns=["Tipo di birra"])  # Feature set
y = df["Tipo di birra"]  # Target

# ✂️ 5. Divisione in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 6. Addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📊 7. Valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# 💾 8. Salvare il modello addestrato
joblib.dump(model, "random_forest_beer_model.pkl")

# 📝 9. Loggare le metriche del training
logging.info(f"Training completato - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

print("✔ Training completato con successo. Log salvato in logs/training_log.txt")