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

# Convertire tutte le colonne in float per evitare problemi con valori mancanti
X = X.astype(float)

# ✂️ 5. Divisione in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏆 6. MLflow Setup
mlflow.set_tracking_uri("file:./mlruns")  # Salva localmente gli artefatti nella directory "mlruns"
mlflow.set_experiment("BeerMatch_Model_Tracking")

# Creare un esempio di input
input_example = X_test.iloc[:1]  # Un esempio reale

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

    # 🔑 9. Creare la firma del modello
    signature = infer_signature(X_train, y_pred)  # Firma basata sui dati di input e predizioni

    # 💾 10. Salvare il modello addestrato
    joblib.dump(model, "random_forest_beer_model.pkl")

    # 📡 11. Logging su MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 📝 12. Loggare le metriche nel file di log
    logging.info(f"Training completato - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # 🚀 13. Salvare il modello su MLflow con firma e esempio
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="beer_match_model",
        signature=signature,
        input_example=input_example
    )

print("✔ Training completato con successo. Log salvato in logs/training_log.txt e MLflow.")