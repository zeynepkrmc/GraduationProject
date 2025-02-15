import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import precision_score, f1_score, recall_score #######3

# Veri yükleme
dataset = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Kategorik sütunları encode etme
label_encoders = {}
categorical_columns = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
for col in categorical_columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Sonuç değişkenini encode etme
label_encoder = LabelEncoder()
dataset['Outcome Variable'] = label_encoder.fit_transform(dataset['Outcome Variable'])

# Özellikleri ve etiketleri ayırma
X = dataset.drop('Outcome Variable', axis=1)
y = dataset['Outcome Variable']

# Normalize edilmesi gereken sütunları ölçeklendirme
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE ile veri dengeleme
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Modellerin tanımlanması
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Linear Regression": LinearRegression()
}

# Modelleri eğitme ve değerlendirme
for model_name, model in models.items():
    print(f"\nModel: {model_name}")
    
    # Modeli eğitme
    model.fit(X_train, y_train)
    
    # Tahmin yapma
    y_pred = model.predict(X_test)
    
    # Doğruluk ve sınıflandırma raporu
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    else:
        # For linear regression, use R2 score as the performance metric
        mse = mean_squared_error(y_test, y_pred)
        r2 = model.score(X_test, y_test)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")

    # Eğer model Linear Regression ise sınıflandırma için yuvarlama yap
    if model_name == "Linear Regression":
        # Sürekli değerleri 0 veya 1'e çevir
        y_pred_class = np.round(y_pred)

        # Accuracy, Precision, Recall, F1-score hesapla
        accuracy = accuracy_score(y_test, y_pred_class)
        precision = precision_score(y_test, y_pred_class, average='weighted')
        recall = recall_score(y_test, y_pred_class, average='weighted')  # Recall 
        f1 = f1_score(y_test, y_pred_class, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")  # Recall metriği burada da eklendi
        print(f"F1 Score: {f1:.4f}")

        # Scatter plot for Linear Regression (predicted vs actual values)
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal fit (y=x)')
        plt.title(f'{model_name} - Predicted vs Actual')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.legend()
        plt.show()

    # Confusion Matrix for classifiers only
    if hasattr(model, 'predict_proba'):
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
        # ROC Curve for classifiers only
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print(f"{model_name} is not a classifier, so confusion matrix and ROC curve are not available.")
# Scatter plot for Linear Regression (predicted vs actual values)
if model_name == "Linear Regression":
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal fit (y=x)')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.legend()
    plt.show()

# En iyi modeli seçip kaydetme (örneğin: Random Forest)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Tüm modeller değerlendirildi ve en iyi model kaydedildi.")
