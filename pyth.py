import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Set page title and layout
st.set_page_config(page_title="Decision Tree Predictor", layout="wide")
st.title("Decision Tree Classifier with Two Features")

# Sidebar for user inputs
st.sidebar.header("Model Input Parameters")


# Load the saved model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r"decision_tree_model.joblib")
        return model
    except FileNotFoundError:
        st.error("Model file 'decision_tree_model.joblib' not found. Please ensure it's in the same directory.")
        st.stop()


model = load_model()

# Check if model has the expected attributes
if not hasattr(model, 'tree_') or not hasattr(model, 'classes_'):
    st.error("The loaded model doesn't appear to be a valid scikit-learn Decision Tree classifier.")
    st.stop()

# Get feature names (assuming the model was trained with feature names)
try:
    feature_names = model.feature_names_in_
    if len(feature_names) != 2:
        st.error("This app is designed for models with exactly 2 features.")
        st.stop()
except AttributeError:
    feature_names = ['Feature 1', 'Feature 2']
    st.warning("Using default feature names as the model doesn't have feature names stored.")

# Get class names
class_names = [str(cls) for cls in model.classes_]

# User input for features
feature1 = st.sidebar.slider(
    label=feature_names[0],
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

feature2 = st.sidebar.slider(
    label=feature_names[1],
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

# Make prediction
input_data = np.array([[feature1, feature2]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display results
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Class", prediction[0])

with col2:
    st.metric("Confidence", f"{np.max(prediction_proba) * 100:.1f}%")

# Show probability distribution
st.subheader("Class Probabilities")
fig, ax = plt.subplots()
ax.bar(class_names, prediction_proba[0])
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)
st.pyplot(fig)

# Visualize the decision tree
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(
    model,
    filled=True,
    feature_names=feature_names,
    class_names=class_names,
    ax=ax,
    proportion=True,
    rounded=True
)
st.pyplot(fig)

# Decision boundary visualization
st.subheader("Decision Boundary")
fig, ax = plt.subplots()

# Create a mesh grid
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict for each point in the mesh grid
try:
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Convert string predictions to numerical if needed
    if isinstance(Z[0], str):
        unique_classes = np.unique(Z)
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        Z = np.array([class_mapping[z] for z in Z])

    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    if np.isfinite(Z).all():  # Check for finite values
        contour = ax.contourf(xx, yy, Z, alpha=0.4, levels=len(np.unique(Z)))

        # Add colorbar if we have numerical classes
        if not isinstance(prediction[0], str):
            plt.colorbar(contour, ax=ax, label='Class')

        ax.scatter(feature1, feature2, c='red', s=100, edgecolor='k', label='Current Input')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Could not plot decision boundary due to invalid values in predictions")
except Exception as e:
    st.error(f"Error plotting decision boundary: {str(e)}")
    st.write("This often happens when the model returns non-numeric predictions.")
    st.write("Prediction sample:", model.predict(np.array([[5.0, 5.0]])))
    st.write("Prediction type:", type(model.predict(np.array([[5.0, 5.0]]))[0]))

# Model information
st.sidebar.header("Model Information")
st.sidebar.write(f"Number of classes: {len(model.classes_)}")
st.sidebar.write(f"Tree depth: {model.get_depth()}")
st.sidebar.write(f"Number of leaves: {model.get_n_leaves()}")