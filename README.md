# Disease Prediction System
Machine Learning cannot independently diagnose patients but can serve as a valuable 
educational tool for medical students. Traditional diagnostic training often lacks interactive, 
data-driven approaches that allow students to practice clinical reasoning in a controlled 
environment. This project aims to develop a web-based platform where medical students can 
engage with ML-powered simulations, enhancing their diagnostic skills through real-world 
case scenarios. By leveraging ML and machine learning, the system provides students with 
dynamic, data-driven feedback, helping them refine their decision-making abilities in a 
structured and interactive manner.
# Features
- Machine learning models trained on patient symptom data
- Data preprocessing including label encoding and scaling
- Multiple classification models (Random Forest, Decision Tree, Logistic Regression, SVM, Naive Bayes, and Linear Regression)
- Data balancing using SMOTE
- The best-performing model (Random Forest) is saved for predictions
- Web-based interface using Flask
# Usage
- The web interface allows users to input symptoms, demographic details, and health indicators.
- The system preprocesses the input, applies the trained model, and returns a prediction.
- Model performance is evaluated using accuracy metrics and visualizations.
# Machine Learning Process
1- Load the dataset and preprocess categorical variables using label encoding.
2- Scale numerical features using MinMaxScaler.
3- Split data into training and testing sets.
4- Apply SMOTE for class balancing.
5- Train multiple models and evaluate their performance.
6- Save the best-performing model (Random Forest) for predictions.
# Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- Seaborn, Matplotlib (for visualization)
- Joblib (for model persistence)
# Licence
- This project is licensed under the MIT License.
