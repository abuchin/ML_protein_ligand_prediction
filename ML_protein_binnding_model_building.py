# Standard libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Get ML libraries
from transformers import AutoTokenizer, AutoModel
import torch

# data manipulation
import pickle

# xgboost
from xgboost import XGBClassifier

# import Py-Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# parallel compute
from joblib import parallel_backend


# save requirements
!pip freeze > requirements_model_build.txt



# all runs
model_reports_runs = {}



# reload combined data
shuffled_data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/combined_data.csv')

# load protein embeddings
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/protein_embeddings.pkl', 'rb') as f:
    protein_embeddings = pickle.load(f)

# load ligand embeddings
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligang_data.pkl', 'rb') as f:
    ligang_data_all = pickle.load(f)



# dictionary to store model results per run
model_reports = {}



# create features and targets for our values
features = []
target = []
taget_binary = []

# whole dataset
n_all=len(shuffled_data)
n_range = 500000

# form training dataset
for j in range(n_range):

  # We take only these values that are present both in protein and ligand embeddings
  if shuffled_data['UniProt_ID'].iloc[j] in protein_embeddings.keys():
    if int(shuffled_data['pubchem_cid'].iloc[j]) in ligang_data_all.keys():

      # concatenate: protein + ligand features
      features.append(list(protein_embeddings[shuffled_data['UniProt_ID'].iloc[j]])+
                      ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['Fingerprint']+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['MolecularWeight']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['PolarSurfaceArea']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['NumRotatableBonds']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['NumHDonors']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['NumHAcceptors']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['NumAromaticRings']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['FractionCSP3']]+
                      [ligang_data_all[int(shuffled_data['pubchem_cid'].iloc[j])]['BertzComplexity']]
                      )

      # safe kiba_score
      target.append(shuffled_data['kiba_score'].iloc[j])
      taget_binary.append(shuffled_data['bound'].iloc[j])



# Create training dataframes
X = pd.DataFrame(features)
y = pd.Series(target)
y_binary = pd.Series(taget_binary)



class LogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, test_size=0.2, random_state=42):
        self.model = LogisticRegression(penalty=penalty, C=C, random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X, y):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self):
        """Train the model on the training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions on the test set."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate(self):
        """Evaluate the model's performance."""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)

        print('Logistic Regression Evaluation')
        print(f"Accuracy: {accuracy}")
        print(report)
        return accuracy, report, cm

    def plot_confusion_matrix(self, cm, labels=['Not Bound', 'Bound']):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix - Logistic Regression")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()



# Initialize the Logistic Regression model
lr_model = LogisticRegressionModel(penalty='l2', C=1.0, test_size=0.2, random_state=42)
# Split the data
lr_model.split_data(X, y_binary)
# Train the model
lr_model.train()
# Make predictions
lr_model.predict()
# Evaluate the model
accuracy, report, cm = lr_model.evaluate()
# Plot the confusion matrix
lr_model.plot_confusion_matrix(cm)

# add reports
model_reports['Logistic Regression'] = [accuracy, report, cm]



# save the model
filename = '/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Models/logistic_regression_model_500K.sav'
pickle.dump(lr_model, open(filename, 'wb'))




class RandomForestModel:

  '''This code describes Random forest classifier re-written in Object-Oriented way'''
  def __init__(self, n_estimators=100, test_size=0.2, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.test_size = test_size
        self.random_state = random_state

  def split_data(self, X, y):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )

  def train(self):
        """Train the model on the training data."""
        self.model.fit(self.X_train, self.y_train)

  def predict(self):
        """Make predictions on the test set."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

  def evaluate(self):
        """Evaluate the model's performance."""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)

        print('Random Forest Evaluation')
        print(f"Accuracy: {accuracy}")
        print(report)
        return accuracy, report, cm

  def plot_confusion_matrix(self, cm, labels=['Not Bound', 'Bound']):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix - Random Forest")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

  def plot_feature_importances(self):
        """Plot feature importances of the model."""
        if hasattr(self.model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.feature_importances_, '.')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance (Random Forest)')
            plt.show()
        else:
            print("Model does not have feature importances attribute.")



# Initialize the Random Forest model
rf_model = RandomForestModel(n_estimators=100, test_size=0.2, random_state=42)
# Split the data
rf_model.split_data(X, y_binary)
# Train the model
rf_model.train()
# Make predictions
rf_model.predict()
# Evaluate the model
accuracy, report, cm = rf_model.evaluate()
# Plot the confusion matrix
rf_model.plot_confusion_matrix(cm)

# add reports
model_reports['Random Forest'] = [accuracy, report, cm]



# Plot feature importances (optional)
rf_model.plot_feature_importances()



# save the model
filename = '/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Models/random_forest_500K.sav'
pickle.dump(rf_model, open(filename, 'wb'))



class XgBoost_ClassificationModel:

  '''This code describes using XbGoost classifier re-written in Object-Oriented way'''

    def __init__(self, model, test_size=0.2, random_state=42):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X, y):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self):
        """Train the model on the training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions on the test set."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate(self):
        """Evaluate the model's performance."""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)

        print('Model Evaluation')
        print(f"Accuracy: {accuracy}")
        print(report)
        return accuracy, report, cm

    def plot_confusion_matrix(self, cm, labels=['Not Bound', 'Bound']):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix - XGBoost")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()



# initialize
xgb_model = XgBoost_ClassificationModel(XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1))
# Split the data
xgb_model.split_data(X, y_binary)
# Train the model
xgb_model.train()
# Make predictions
xgb_model.predict()
# Evaluate the model
accuracy, report, cm = xgb_model.evaluate()
# Plot the confusion matrix
xgb_model.plot_confusion_matrix(cm)

# add report
model_reports['XGBoost'] = [accuracy, report, cm]



# save results
filename = '/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Models/xgboost_model_500K.sav'
pickle.dump(xgb_model, open(filename, 'wb'))



class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MLPClassifier:
    def __init__(self, input_size, hidden_size=256, output_size=2, batch_size=32, lr=0.001, epochs=10):
        self.model = MLP(input_size, hidden_size, output_size)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def split_data(self, X, y):
        """Split data into training and testing sets and create dataloaders."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        self.train_dataset = MyDataset(X_train, y_train)
        self.test_dataset = MyDataset(X_test, y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.y_test = y_test  # Store y_test for evaluation

    def train(self):
        """Train the MLP model."""
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        print('MLP Evaluation')
        print(f"Accuracy: {accuracy}")
        print(report)
        return accuracy, report, cm

    def save(self, path):
        """Save the model's state dictionary."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load the model's state dictionary."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def plot_confusion_matrix(self, cm, labels=['Not Bound', 'Bound']):
        """Plot the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix - MLP (PyTorch)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


# Initialize the MLP classifier with relevant parameters
input_size = X.shape[1]
mlp_classifier = MLPClassifier(input_size=input_size, hidden_size=256, output_size=2, batch_size=32, lr=0.001, epochs=10)
# Split the data
mlp_classifier.split_data(X, y_binary)
# Train the model
mlp_classifier.train()
# Evaluate the model
accuracy, report, cm = mlp_classifier.evaluate()
# Plot the confusion matrix
mlp_classifier.plot_confusion_matrix(cm)

# Save model report
model_reports['MLP'] = [accuracy, report, cm]



# Save MLP model
mlp_classifier.save('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Models/mlp_model_500K.pth')



# save model reports into dictionary
model_reports_runs['10K'] = model_reports.copy()



# Extract accuracy scores for each model
accuracy_scores = [model_reports_runs['10K']['Logistic Regression'][0],
                   model_reports_runs['10K']['Random Forest'][0],
                   model_reports_runs['10K']['XGBoost'][0],
                   model_reports_runs['10K']['MLP'][0]]

# Model names
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP']

# Create the bar plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.bar(model_names, accuracy_scores)
# added baseline
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline (0.5)')  # Add baseline at 0.5
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Model Accuracy Comparison (' + str('10K') +str(' samples)'))
plt.show()




# save
model_reports_runs['50K']=model_reports.copy()



# Extract accuracy scores for each model
accuracy_scores = [model_reports_runs['50K']['Logistic Regression'][0],
                   model_reports_runs['50K']['Random Forest'][0],
                   model_reports_runs['50K']['XGBoost'][0],
                   model_reports_runs['50K']['MLP'][0]]

# Model names
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP']

# Create the bar plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.bar(model_names, accuracy_scores)
# added baseline
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline (0.5)')  # Add baseline at 0.5
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Model Accuracy Comparison (' + str('50K') +str(' samples)'))
plt.show()



# save
model_reports_runs['500K']=model_reports.copy()



# Extract accuracy scores for each model
accuracy_scores = [model_reports_runs['500K']['Logistic Regression'][0],
                   model_reports_runs['500K']['Random Forest'][0],
                   model_reports_runs['500K']['XGBoost'][0],
                   model_reports_runs['500K']['MLP'][0]]

# Model names
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP']

# Create the bar plot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.bar(model_names, accuracy_scores)
# added baseline
plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline (0.5)')  # Add baseline at 0.5
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Model Accuracy Comparison (' + str('500K') +str(' samples)'))
plt.show()



# Create a DataFrame for plotting
data = []
for dataset_size, model_reports in model_reports_runs.items():
  for model, report in model_reports.items():
    # Accuracy is the first element
      data.append([dataset_size, model, report[0]])

df = pd.DataFrame(data, columns=['Dataset Size', 'Model', 'Accuracy'])

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='Dataset Size', y='Accuracy', hue='Model', data=df, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Training data')
plt.ylim([0, 1])  # Set y-axis limit
plt.show()



