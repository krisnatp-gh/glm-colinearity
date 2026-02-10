import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)

# Page configuration
st.set_page_config(page_title="Regularization Demo", layout="wide")


# Helper Functions

@st.cache_data
def load_data():
    """Load the wages dataset with caching"""
    try:
        df = pd.read_csv("wages.csv")
        return df
    except FileNotFoundError:
        st.error("Error: wages.csv not found. Please ensure it's in the same directory as app.py")
        st.stop()


def prepare_features(df):
    """Separate features and target, identify column types"""
    X = df.drop(columns="WAGE")
    y = df["WAGE"]
    numerical_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    return X, y, numerical_cols, categorical_cols


def build_pipeline(model_name, alpha_value, numerical_cols, categorical_cols):
    """Build sklearn pipeline with preprocessing and model"""
    # Preprocessing
    preprocess = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(drop="first", sparse_output=False,
                                    handle_unknown="ignore"), categorical_cols),
            ("robust", RobustScaler(), numerical_cols)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    preprocess.set_output(transform="pandas")

    # Model selection
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha_value, random_state=42)
    elif model_name == "Lasso":
        model = Lasso(alpha=alpha_value, random_state=42, max_iter=10000)

    return Pipeline([("preprocessing", preprocess), ("modeling", model)])


def compute_metrics(pipeline, X_train, X_test, y_train, y_test):
    """Compute all evaluation metrics for train and test sets"""
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    return {
        "rmse_train": root_mean_squared_error(y_train, y_train_pred),
        "rmse_test": root_mean_squared_error(y_test, y_test_pred),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "mape_train": mean_absolute_percentage_error(y_train, y_train_pred),
        "mape_test": mean_absolute_percentage_error(y_test, y_test_pred),
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred)
    }


def compute_coefficient_variability(pipeline, X_train, y_train):
    """
    Use RepeatedKFold CV to compute coefficient variability.
    Scale coefficients by feature std for fair comparison.
    """
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv, return_estimator=True, n_jobs=-1
    )

    feature_names = pipeline[:-1].get_feature_names_out()

    # Extract scaled coefficients from each CV fold
    coefs = pd.DataFrame(
        [
            est[-1].coef_ * est[:-1].transform(X_train.iloc[train_idx]).std(axis=0)
            for est, (train_idx, _) in zip(cv_results["estimator"],
                                           cv.split(X_train, y_train))
        ],
        columns=feature_names
    )

    return coefs, feature_names


def create_coefficient_plot(coef_df, model_name, alpha_value):
    """Create stripplot + boxplot showing coefficient variability"""
    # Detect extreme outliers due to multicollinearity
    max_abs_coef = coef_df.abs().max().max()
    median_abs_coef = coef_df.abs().median().median()
    has_extreme_outliers = max_abs_coef > 1e6 or (median_abs_coef > 0 and max_abs_coef / median_abs_coef > 1e6)

    # Compute IQR bounds for extreme outlier detection
    if has_extreme_outliers:
        all_values = coef_df.values.flatten()
        q1 = np.percentile(all_values, 20)
        q3 = np.percentile(all_values, 87)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clipped = True
    else:
        clipped = False
        lower_bound = upper_bound = None

    # Sort by absolute median coefficient
    median_coefs = coef_df.median().abs().sort_values(ascending=True)
    coef_df_sorted = coef_df[median_coefs.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.stripplot(data=coef_df_sorted, orient="h", palette="dark:k",
                  alpha=0.5, ax=ax)
    sns.boxplot(data=coef_df_sorted, orient="h", color="cyan",
                saturation=0.5, whis=10, ax=ax)
    ax.axvline(x=0, color=".5", linestyle="--", linewidth=1)

    # Clip x-axis and show bounds if extreme outliers detected
    if clipped:
        ax.set_xlim(lower_bound, upper_bound)
        ax.axvline(x=lower_bound, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.axvline(x=upper_bound, color="red", linestyle=":", linewidth=1.5, alpha=0.7)

        max_val = all_values.max()
        min_val = all_values.min()

        # Annotate extreme positive outlier (right side, at bottom of plot)
        if max_val > upper_bound:
            ax.text(
                upper_bound, len(coef_df.columns) - 0.5,
                f"  max = {max_val:.2e}\n  (off chart →)",
                fontsize=9, color="red", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8)
            )

        # Annotate extreme negative outlier (left side, at bottom of plot)
        if min_val < lower_bound:
            ax.text(
                lower_bound, len(coef_df.columns) - 0.5,
                f"← min = {min_val:.2e}  \n(off chart)  ",
                fontsize=9, color="red", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8)
            )

    ax.set_xlabel("Coefficient importance (scaled by feature std)", fontsize=11)

    if alpha_value:
        title = f"Coefficient Variability: {model_name} (α={alpha_value:.2e})"
    else:
        title = f"Coefficient Variability: {model_name}"

    if clipped:
        title += "\n⚠ Extreme instabilities clipped for visualization"

    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig, has_extreme_outliers, max_abs_coef, lower_bound, upper_bound


def train_model(X_train, X_test, y_train, y_test, model_name,
                alpha_value, numerical_cols, categorical_cols):
    """Train model and compute all metrics and visualizations"""
    pipeline = build_pipeline(model_name, alpha_value, numerical_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    metrics = compute_metrics(pipeline, X_train, X_test, y_train, y_test)
    coef_df, feature_names = compute_coefficient_variability(pipeline, X_train, y_train)

    return pipeline, metrics, coef_df, feature_names


# Main Application

def main():
    # Title and Description
    st.title("📊 Regularization & Multicollinearity Demo")
    st.markdown("""
    This interactive app demonstrates how **regularization** (Ridge/Lasso) reduces
    **multicollinearity** by stabilizing coefficient estimates.
    """)

    # Sample Data Section
    st.subheader("📋 Sample Data")
    df = load_data()
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # ML Configuration Section
    st.subheader("⚙️ ML Configuration")

    col1, col2 = st.columns([1, 2])
    with col1:
        model_name = st.selectbox("Select Model",
                                  ["Linear Regression", "Ridge", "Lasso"])

    with col2:
        if model_name in ["Ridge", "Lasso"]:
            alpha_log = st.slider(
                f"Alpha (α) for {model_name}",
                min_value=-5.0, max_value=5.0, value=1.0, step=0.5,
                format="10^%.1f",
                help="Logarithmic scale: 10^x"
            )
            alpha_value = 10 ** alpha_log
            st.caption(f"Actual α = {alpha_value:.6f}")
        else:
            alpha_value = None
            st.info("Linear Regression has no regularization parameter")

    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Train Button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        X, y, numerical_cols, categorical_cols = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        with st.spinner("Training model and computing coefficient variability..."):
            try:
                pipeline, metrics, coef_df, feature_names = train_model(
                    X_train, X_test, y_train, y_test,
                    model_name, alpha_value, numerical_cols, categorical_cols
                )

                # Store in session state
                st.session_state.model_trained = True
                st.session_state.metrics = metrics
                st.session_state.coef_df = coef_df
                st.session_state.model_name = model_name
                st.session_state.alpha_value = alpha_value

                st.success("✅ Training complete!")
            except Exception as e:
                st.error(f"Error during training: {e}")
                st.exception(e)

    # Results Display (Conditional)
    if st.session_state.model_trained:
        st.subheader("📈 Model Performance")

        # Metrics scorecards (4 columns)
        col1, col2, col3, col4 = st.columns(4)
        metrics = st.session_state.metrics

        with col1:
            st.metric("RMSE (Train)", f"{metrics['rmse_train']:.4f}")
            st.metric("RMSE (Test)", f"{metrics['rmse_test']:.4f}")

        with col2:
            st.metric("MAE (Train)", f"{metrics['mae_train']:.4f}")
            st.metric("MAE (Test)", f"{metrics['mae_test']:.4f}")

        with col3:
            st.metric("MAPE (Train)", f"{metrics['mape_train']:.4f}")
            st.metric("MAPE (Test)", f"{metrics['mape_test']:.4f}")

        with col4:
            st.metric("R² (Train)", f"{metrics['r2_train']:.4f}")
            st.metric("R² (Test)", f"{metrics['r2_test']:.4f}")

        # Coefficient variability chart
        st.subheader("🎯 Coefficient Variability")
        fig, has_extreme_outliers, max_coef, lower_bound, upper_bound = create_coefficient_plot(
            st.session_state.coef_df,
            st.session_state.model_name,
            st.session_state.alpha_value
        )
        st.pyplot(fig)

        # Warning about extreme multicollinearity
        if has_extreme_outliers:
            st.warning(
                f"🔴 **Severe multicollinearity detected!** "
                f"Some coefficients reached ±{max_coef:.2e} (extreme instability). "
                f"This demonstrates why regularization is needed. "
            )

        # Educational explanation
        st.markdown("""
        **How to interpret:**
        - **Wide spread** = High variability = Multicollinearity present
        - **Narrow spread** = Low variability = Regularization working
        - Each point represents coefficient from one CV fold (25 total)
        """)


if __name__ == "__main__":
    main()
