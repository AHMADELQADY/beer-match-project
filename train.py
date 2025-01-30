import logging
import os
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature

# 📂 Configura la directory per il tracking di MLflow
tracking_dir = "./mlruns"  # Directory relativa per i dati di tracking
artifact_dir = "./mlartifacts"  # Directory relativa per gli artifact
os.makedirs(tracking_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")

# 📂 Creazione della cartella per i log
log_dir = "./logs"
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
joblib.dump(label_encoders, os.path.join(artifact_dir, "label_encoders.pkl"))

# 🔄 4. Separazione delle feature e del target
X = df.drop(columns=["Tipo di birra"])  # Feature set
y = df["Tipo di birra"]  # Target

# Convertire tutte le colonne in float per evitare problemi con valori mancanti
X = X.astype(float)

# ✂️ 5. Divisione in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏆 6. MLflow Setup
mlflow.set_experiment("BeerMatch_Model_Tracking")

with mlflow.start_run():
    # 🤖 7. Addestramento del modello
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 📊 8. Valutazione del modello
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Creare un esempio di input e inferire la firma
    input_example = X_test.iloc[:1]  # Un esempio reale
    signature = infer_signature(X_train, model.predict(X_train))  # Firma basata sui dati di input e output

    # 💾 9. Salvare il modello addestrato
    model_path = os.path.join(artifact_dir, "random_forest_beer_model.pkl")
    joblib.dump(model, model_path)

    # 📡 10. Logging su MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 📝 11. Loggare le metriche nel file di log
    logging.info(f"Training completato - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # 🚀 12. Salvare il modello su MLflow con firma e esempio
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="beer_match_model",
        signature=signature,
        input_example=input_example
    )

print("✔ Training completato con successo. Log salvato in logs/training_log.txt e MLflow.")