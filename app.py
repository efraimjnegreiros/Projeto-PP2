# Suposto app.py

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset (ajuste o caminho do arquivo conforme necessário)
df = pd.read_csv('./autism_screening.csv')  # Substitua 'seu_arquivo.csv' pelo caminho correto do seu arquivo

# Função para treinar o modelo com as 18 variáveis especificadas
def treinar_modelo(df):
    # Limpeza dos dados

    # Remover valores nulos na coluna 'age'
    df = df.dropna(subset=['age'])

    # Remover linhas com "?" (valores inválidos) em qualquer coluna
    df = df[~df.isin(['?']).any(axis=1)]

    # Criar o LabelEncoder
    label_encoder = LabelEncoder()

    # Aplicar o LabelEncoder para todas as colunas que contêm valores categóricos
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Balancear as classes na coluna 'Class/ASD'
    autistas = df[df['Class/ASD'] == 1]  # Supondo que '1' represente "YES"
    nao_autistas = df[df['Class/ASD'] == 0]  # Supondo que '0' represente "NO"

    # Encontrar o número mínimo de instâncias entre as duas classes
    min_count = min(len(autistas), len(nao_autistas))

    # Balancear as classes, fazendo undersampling
    autistas_balanced = autistas.sample(min_count, random_state=42)
    nao_autistas_balanced = nao_autistas.sample(min_count, random_state=42)

    # Concatenar as classes balanceadas
    df_balanced = pd.concat([autistas_balanced, nao_autistas_balanced])

    # Embaralhar os dados balanceados
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Selecionar as 18 variáveis para treinar o modelo
    X = df_balanced[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                     'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'result', 'age_desc', 'relation']]  # Usando as 18 variáveis
    y = df_balanced['Class/ASD']  # Variável alvo

    # Dividindo os dados em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizando os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Regressão Logística
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    # Previsões no conjunto de teste
    y_pred_logreg = logreg.predict(X_test_scaled)

    # Acurácia
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    print(f'Acurácia (Logistic Regression): {accuracy_logreg:.4f}')

    # Relatório de classificação
    print('Relatório de Classificação (Logistic Regression):')
    print(classification_report(y_test, y_pred_logreg))

    # Métricas adicionais
    precision = precision_score(y_test, y_pred_logreg)
    recall = recall_score(y_test, y_pred_logreg)
    f1 = f1_score(y_test, y_pred_logreg)
    roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1])

    print(f'\nPrecisão: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')

    return logreg, scaler

# Treinando o modelo
logreg, scaler = treinar_modelo(df)

# Função para fazer previsões com o modelo treinado
def prever_autismo(input_data):
    input_scaled = scaler.transform([input_data])
    prediction = logreg.predict(input_scaled)
    probability = logreg.predict_proba(input_scaled)[0][1]
    return prediction, probability

# Streamlit UI
st.title("Previsão de Autismo")
st.write("Por favor, insira os valores abaixo para as variáveis:")

# Coleta dos dados via input (aqui estamos coletando 18 variáveis)
A1_Score = st.number_input("A1_Score", min_value=0, max_value=1, step=1)
A2_Score = st.number_input("A2_Score", min_value=0, max_value=1, step=1)
A3_Score = st.number_input("A3_Score", min_value=0, max_value=1, step=1)
A4_Score = st.number_input("A4_Score", min_value=0, max_value=1, step=1)
A5_Score = st.number_input("A5_Score", min_value=0, max_value=1, step=1)
A6_Score = st.number_input("A6_Score", min_value=0, max_value=1, step=1)
A7_Score = st.number_input("A7_Score", min_value=0, max_value=1, step=1)
A8_Score = st.number_input("A8_Score", min_value=0, max_value=1, step=1)
A9_Score = st.number_input("A9_Score", min_value=0, max_value=1, step=1)
A10_Score = st.number_input("A10_Score", min_value=0, max_value=1, step=1)

gender = st.number_input("Gender (1 = Masculino, 0 = Feminino)", min_value=0, max_value=1, step=1)
ethnicity = st.number_input("Ethnicity (Código numérico)", min_value=0, max_value=5, step=1)
jundice = st.number_input("Jundice (1 = Sim, 0 = Não)", min_value=0, max_value=1, step=1)
austim = st.number_input("Autismo (1 = Sim, 0 = Não)", min_value=0, max_value=1, step=1)
contry_of_res = st.number_input("Country of Residence (Código numérico)", min_value=1, max_value=5, step=1)
used_app_before = st.number_input("Usou aplicativo antes? (1 = Sim, 0 = Não)", min_value=0, max_value=1, step=1)
result = st.number_input("Resultado do teste", min_value=0, max_value=1, step=1)
age_desc = st.number_input("Grupo etário (0: 0-5, 1: 6-10, 2: 11-18, 3: 18+)", min_value=0, max_value=3, step=1)
relation = st.number_input("Tipo de relação (0 = Self, 1 = Parent, etc.)", min_value=0, max_value=1, step=1)

# Preparando os dados de entrada para o modelo
input_data = [A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score,
              gender, ethnicity, jundice, austim, contry_of_res, used_app_before, result, age_desc, relation]

# Quando o botão é pressionado, fazer a previsão
if st.button('Fazer Previsão'):
    prediction, probability = prever_autismo(input_data)

    if prediction == 1:
        st.write("O indivíduo é autista.")
    else:
        st.write("O indivíduo não é autista.")

    st.write(f"Probabilidade de ser autista: {probability:.4f}")
