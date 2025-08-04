from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

y_binary = (y['num'] > 0).astype(int).values.ravel()

variables = ['age', 'sex', 'cp', 'trestbps', 'exang']
df = X[variables].copy()
df['maladie'] = y_binary

df = df.dropna()

def discretize_age(age):
    if age < 45:
        return 0
    elif age < 65:
        return 1
    else:
        return 2

def discretize_bp(bp):
    if bp < 120:
        return 0
    elif bp < 140:
        return 1
    else:
        return 2

df['age_cat'] = df['age'].apply(discretize_age)
df['bp_cat'] = df['trestbps'].apply(discretize_bp)
df = df[['age_cat', 'sex', 'cp', 'bp_cat', 'exang', 'maladie']]

X_data = df.drop('maladie', axis=1)
y_data = df['maladie']

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
)

def train(X, y):
    priors = {}
    likelihoods = {}
    
    n_total = len(y)
    for c in [0, 1]:
        priors[c] = (y == c).sum() / n_total
    
    features = X.columns.tolist()
    
    for feature in features:
        likelihoods[feature] = {0: {}, 1: {}}
        
        for c in [0, 1]:
            X_c = X[y == c]
            n_c = len(X_c)
            
            for val in X[feature].unique():
                count = (X_c[feature] == val).sum()
                likelihoods[feature][c][val] = (count + 1) / (n_c + len(X[feature].unique()))
    
    return priors, likelihoods, features

def predict(X, priors, likelihoods, features):
    predictions = []
    probas = []
    
    for idx in X.index:
        probs = {}
        
        for c in [0, 1]:
            prob = priors[c]
            
            for feature in features:
                val = X.loc[idx, feature]
                if val in likelihoods[feature][c]:
                    prob *= likelihoods[feature][c][val]
                else:
                    prob *= 1e-6
            
            probs[c] = prob
        
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        probas.append([probs[0], probs[1]])
        predictions.append(1 if probs[1] > 0.5 else 0)
    
    return np.array(predictions), np.array(probas)

priors, likelihoods, features = train(X_train, y_train)

print(f"P(Pas de maladie) = {priors[0]:.3f}")
print(f"P(Maladie) = {priors[1]:.3f}")

y_pred, y_proba = predict(X_test, priors, likelihoods, features)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.2%}")
print("\nMatrice de confusion:")
print(cm)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax1 = axes[0]
probas_malades = y_proba[y_test == 1, 1]
probas_sains = y_proba[y_test == 0, 1]

ax1.hist(probas_sains, bins=15, alpha=0.5, label='Sains', color='green')
ax1.hist(probas_malades, bins=15, alpha=0.5, label='Malades', color='red')
ax1.axvline(x=0.5, color='black', linestyle='--')
ax1.set_xlabel('Probabilité de maladie')
ax1.set_ylabel('Nombre')
ax1.legend()

ax2 = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Prédiction')
ax2.set_ylabel('Réalité')
ax2.set_xticklabels(['Sain', 'Malade'])
ax2.set_yticklabels(['Sain', 'Malade'])

plt.tight_layout()
plt.show()

nouveau = pd.DataFrame({
    'age_cat': [2],
    'sex': [1],
    'cp': [4],
    'bp_cat': [2],
    'exang': [1]
})

pred, proba = predict(nouveau, priors, likelihoods, features)

print(f"\nNouveau patient:")
print(f"P(maladie) = {proba[0][1]:.2%}")
print(f"Décision: {'Malade' if pred[0] == 1 else 'Sain'}")

def analyze_features():
    for feature in features:
        print(f"\n{feature}:")
        for val in sorted(X_train[feature].unique()):
            if val in likelihoods[feature][1] and val in likelihoods[feature][0]:
                ratio = likelihoods[feature][1][val] / likelihoods[feature][0][val]
                print(f"  {val}: ratio = {ratio:.2f}")

analyze_features()