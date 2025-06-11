import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

DATA_PATH = 'path/to/your/file.csv'
MODEL_PATH = 'path/to/your/model.pkl'
SAVE_DIR = 'shap_output'  # Optional directory to save plots
FEATURES_TO_PLOT = ["indicators"]

# -------------------------------
# Step 2: Load Data & Model
# -------------------------------
df = pd.read_csv(DATA_PATH)
cols = df.columns.drop('label')
X = df[cols]
y = df['label']

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, max_display=20, plot_type='bar')


def plot_dependence(features, shap_values, X, save_dir=None, interaction_index=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for feature in features:
        shap.dependence_plot(
            feature, shap_values, X, interaction_index=interaction_index, show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature}")
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"shap_dependence_{feature}.png"), bbox_inches='tight', dpi=600)
        plt.show()

plot_dependence(FEATURES_TO_PLOT, shap_values, X, save_dir=SAVE_DIR)



