import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸")


@st.cache_resource
def train_models():
    df = sns.load_dataset('iris')
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (clf, acc)

    best_name = max(results, key=lambda n: results[n][1])
    best_model, best_acc = results[best_name]

    return best_model, best_name, best_acc, X.columns.tolist()


model, model_name, accuracy, feature_names = train_models()

st.title("🌸 Iris Species Classifier")
st.write(
    f"Predicts the Iris flower species from petal and sepal measurements. "
    f"Best performing model: **{model_name}** (test accuracy: {accuracy:.2%})."
)

st.subheader("Enter flower measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

if st.button("Predict Species", type="primary"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names,
    )
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted species: **{prediction.capitalize()}**")

st.caption(f"Dataset: Iris (seaborn) · Model: {model_name}")
