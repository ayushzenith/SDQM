import argparse
import json
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def train_regression_model(X_train, y_train, method):
    if method == "linear":
        model = LinearRegression()
    elif method == "ridge":
        model = Ridge()
    elif method == "lasso":
        model = Lasso()
    elif method == "decision_tree":
        model = DecisionTreeRegressor()
    elif method == "random_forest":
        model = RandomForestRegressor()
    elif method == "xgboost":
        model = xgb.XGBRegressor()
    elif method == "svr":
        model = SVR()
    else:
        raise ValueError(f"Unknown method: {method}")
    model.fit(X_train, y_train)
    return model


def get_model_coefficients(model, metrics):
    """
    Get the coefficients and intercept of the model if available.
    """
    result = {}
    if hasattr(model, "coef_"):
        coef = model.coef_
        if isinstance(coef, np.ndarray):
            coef = coef.tolist()
        coef_dict = dict(zip(metrics, coef))
        result["coef"] = coef_dict
    if hasattr(model, "intercept_"):
        intercept = model.intercept_
        if isinstance(intercept, np.ndarray):
            intercept = intercept.tolist()
        else:
            intercept = float(intercept)
        result["intercept"] = intercept
    return result


def get_feature_importances(model, metrics):
    """
    Get the feature importances of the model if available.
    """
    result = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if isinstance(importances, np.ndarray):
            importances = importances.tolist()
        importances_dict = dict(zip(metrics, importances))
        result["feature_importances"] = importances_dict
    return result


def compute_correlations(y_pred, y_test, output_path, method_name, y_label, fold_index=None):
    """
    Calculate the Pearson and Spearman correlation coefficients between the model's predictions and y_test.
    """
    # Calculate Pearson correlation
    pearson_corr, pearson_p_value = pearsonr(y_pred, y_test)
    # Calculate Spearman correlation
    spearman_corr, spearman_p_value = spearmanr(y_pred, y_test)
    # Create a scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, y_test, color="blue")
    # Add the correlation coefficients and p-values text on the plot
    plt.text(
        0.05,
        0.95,
        f"Pearson r = {pearson_corr:.2f}, p = {pearson_p_value:.3e}",
        fontsize=12,
        color="red",
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.05,
        0.90,
        f"Spearman ρ = {spearman_corr:.2f}, p = {spearman_p_value:.3e}",
        fontsize=12,
        color="green",
        transform=plt.gca().transAxes,
    )
    # Add labels and title
    plt.xlabel("Predicted Value")
    plt.ylabel(y_label)
    title = f"{y_label} vs. Predicted ({method_name})"
    if fold_index is not None:
        title += f" - Fold {fold_index + 1}"
    plt.title(title)
    plt.grid(True)
    # Save the plot
    if fold_index is not None:
        plot_file = os.path.join(output_path, f"scatter_plot_{method_name}_fold_{fold_index + 1}.png")
    else:
        plot_file = os.path.join(output_path, f"scatter_plot_{method_name}.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    return {
        "pearson_corr": pearson_corr,
        "pearson_p_value": pearson_p_value,
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p_value,
    }


def plot_residuals(y_pred, y_test, output_path, method_name, y_label, fold_index=None):
    """
    Plot the residuals of the predictions.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, color="green")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residuals")
    title = f"Residuals vs. Predicted ({method_name})"
    if fold_index is not None:
        title += f" - Fold {fold_index + 1}"
    plt.title(title)
    plt.grid(True)
    if fold_index is not None:
        plot_file = os.path.join(output_path, f"residuals_plot_{method_name}_fold_{fold_index + 1}.png")
    else:
        plot_file = os.path.join(output_path, f"residuals_plot_{method_name}.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()


def plot_feature_correlations(df_features, y, output_path, y_label, method="pearson", top_n=None):
    """
    Compute and plot the correlations between each feature and y.
    Also, output a CSV of these correlations.
    """
    if method == "pearson":
        corr_method = lambda x: pearsonr(x, y)[0]
    elif method == "spearman":
        corr_method = lambda x: spearmanr(x, y)[0]
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    # Compute correlations
    corrs = df_features.apply(corr_method)
    corrs_df = pd.DataFrame(corrs, columns=["Correlation"])

    # If top_n is specified, select top N features
    if top_n is not None and top_n < len(corrs_df):
        corrs_df = corrs_df.reindex(corrs_df["Correlation"].abs().sort_values(ascending=False).index)
        corrs_df = corrs_df.iloc[:top_n]

    # Save the correlations to CSV
    correlations_csv = os.path.join(output_path, f"{method}_feature_correlations.csv")
    corrs_df.to_csv(correlations_csv)

    # Adjust figure size
    num_features = len(corrs_df)
    plt.figure(figsize=(max(12, num_features * 0.5), 6))

    # Use smaller font size for annotations
    sns.heatmap(corrs_df.T, annot=True, annot_kws={"size": 8}, cmap="coolwarm", cbar=False, fmt=".2f")
    plt.title(f"{method.capitalize()} Correlations between Features and {y_label}")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    heatmap_file = os.path.join(output_path, f"{method}_correlation_heatmap.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()


def plot_full_correlation_matrix(df, output_path, method="pearson", filename_prefix="full"):
    """
    Compute and plot the full correlation matrix of the DataFrame including features and target variable.
    Also output a CSV file of the correlation matrix.
    """

    # Compute the correlation matrix
    corr_matrix = df.corr(method=method)

    # Save correlation matrix as CSV
    csv_file = os.path.join(output_path, f"{filename_prefix}_{method}_correlation_matrix.csv")
    corr_matrix.to_csv(csv_file)

    # Adjust figure size based on the number of variables
    num_vars = len(corr_matrix)
    plt.figure(figsize=(num_vars * 0.5, num_vars * 0.5))

    # Draw the heatmap
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        annot_kws={"size": 6},
        cmap="coolwarm",
        fmt=".2f",
        cbar=True
    )

    # Adjust annotation colors dynamically for contrast
    for text in ax.texts:  # `ax.texts` contains all annotations
        value = float(text.get_text())
        text_color = "white" if value < -0.7 or value > 0.7 else "black"
        text.set_color(text_color)

    # Add titles and save the heatmap
    plt.title(f"{method.capitalize()} Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_file = os.path.join(output_path, f"{filename_prefix}_{method}_correlation_matrix.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()

def plot_metrics_vs_y(df, y_label, output_path):
    """
    Plot and save each metric against the target variable y, including Pearson and Spearman correlation coefficients.
    """
    metric_plots_path = os.path.join(output_path, "metric_plots")
    os.makedirs(metric_plots_path, exist_ok=True)
    
    for metric in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[metric], df[y_label], color="blue")
        
        # Calculate Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(df[metric], df[y_label])
        spearman_corr, _ = spearmanr(df[metric], df[y_label])
        
        # Add correlation coefficients to the plot
        plt.text(
            0.05,
            0.95,
            f"Pearson r = {pearson_corr:.2f}",
            fontsize=12,
            color="red",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.05,
            0.90,
            f"Spearman ρ = {spearman_corr:.2f}",
            fontsize=12,
            color="green",
            transform=plt.gca().transAxes,
        )
        
        plt.xlabel(metric)
        plt.ylabel(y_label)
        plt.title(f"{metric} vs. {y_label}")
        plt.grid(True)
        plot_file = os.path.join(metric_plots_path, f"{metric}_vs_{y_label}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()


def perform_pca(X, n_components):
    """
    Perform PCA on the input data X.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def combine_variables_with_same_root(df_features, importance_dict):
    """
    Combine variables that share the same root name using the absolute values of regression coefficients or feature importances as weights.
    """
    if not importance_dict:
        raise ValueError("Importance weights are not available for combining variables.")

    root_to_variables = defaultdict(list)
    for var_name in importance_dict.keys():
        root_name = var_name.split("_")[0]
        root_to_variables[root_name].append(var_name)

    df_combined = pd.DataFrame()

    for root_name, var_list in root_to_variables.items():
        weights = np.array([abs(importance_dict[var_name]) for var_name in var_list])
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights.sum()
        variables = df_features[var_list]
        combined_metric = (variables * weights).sum(axis=1)
        df_combined[root_name] = combined_metric

    return df_combined


def run_k_fold(X, y, method, metrics, output_path, y_label, n_splits=5):
    """
    Run k-fold cross-validation and return average Pearson/Spearman correlations.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pearson_corrs = []
    spearman_corrs = []

    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        model = train_regression_model(X_train, y_train, method)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        correlations = compute_correlations(y_pred, y_test, output_path, method, y_label, fold_index)
        pearson_corrs.append(correlations["pearson_corr"])
        spearman_corrs.append(correlations["spearman_corr"])

    pearson_corr_mean = np.mean(pearson_corrs)
    spearman_corr_mean = np.mean(spearman_corrs)
    return pearson_corr_mean, spearman_corr_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, nargs="+", help="Path(s) to the input csv file(s)")
    parser.add_argument("--val_input_path", type=str, nargs="+", help="Path(s) to the validation csv file(s)")
    parser.add_argument("--output_path", type=str, help="Path to the output directory")
    parser.add_argument("--start_column", type=int, help="Column to start regression from")
    parser.add_argument("--y_column", type=int, help="Column to use as target variable")
    parser.add_argument("--last", action="store_true", help="Use the last column as the target variable")
    parser.add_argument(
        "--load_results", action="store_true", help="Load results.json and calculate Pearson coefficient with new data"
    )
    parser.add_argument("--standardize", action="store_true", help="Standardize the input data")
    parser.add_argument("--pca", type=int, default=None, help="Number of principal components to keep")
    parser.add_argument(
        "--method",
        type=str,
        default="linear",
        choices=["linear", "ridge", "lasso", "decision_tree", "random_forest", "xgboost", "svr"],
        help="Regression method to use",
    )
    parser.add_argument("--all_methods", action="store_true", help="Run all regression methods")
    parser.add_argument(
        "--shuffle_split", action="store_true", help="Shuffle and split data into train and validation sets"
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use as test set")
    parser.add_argument("--k_folds", type=int, default=None, help="Number of folds for k-fold cross-validation")
    parser.add_argument(
        "--feature_removal_test",
        action="store_true",
        help="Remove each feature one by one and measure the effect on correlation coefficients",
    )
    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=None,
        help="Remove features whose absolute Pearson correlation with target is below this threshold",
    )
    parser.add_argument("--sequential_test", type=int, default=None, help="Perform sequential feature selection test")
    parser.add_argument("--scaler", type=str, default=None, help="Path to a saved StandardScaler object")
    parser.add_argument("--separately_scale", action="store_true", help="Standardize each csv file separately")
    args = parser.parse_args()

    run_regression(args)


def run_regression(args):
    if args.separately_scale and args.standardize:
        raise ValueError("Cannot specify both separately_scale and standardize options together.")

    if args.last:
        args.y_column = -1
        args.start_column = 1
        end_column = -1
    else:
        end_column = None
    os.makedirs(args.output_path, exist_ok=True)

    # Determine y-axis label based on the last column name of the input CSV file
    sample_df = pd.read_csv(args.input_path[0])
    y_label = sample_df.columns[args.y_column]

    # Read and concatenate all input CSV files
    data_frames = []
    for input_file in args.input_path:
        df = pd.read_csv(input_file)
        if args.separately_scale:
            scaler = StandardScaler()
            df.iloc[:, args.start_column:end_column] = scaler.fit_transform(df.iloc[:, args.start_column:end_column])
            scaler_file = os.path.join(args.output_path, f"scaler_{os.path.basename(input_file)}.pkl")
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
        data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)

    original_metrics = data.columns[args.start_column : end_column].tolist()
    X_with_names = data.iloc[:, args.start_column : end_column]
    X = data.iloc[:, args.start_column : end_column].values
    y = data.iloc[:, args.y_column].values

    df_features = data.iloc[:, args.start_column : end_column]
    df_y = data.iloc[:, args.y_column]

    # Read and concatenate all validation CSV files if provided
    if args.val_input_path:
        val_data_frames = []
        for val_input_file in args.val_input_path:
            val_df = pd.read_csv(val_input_file)
            if args.separately_scale:
                scaler = StandardScaler()
                val_df.iloc[:, args.start_column:end_column] = scaler.fit_transform(val_df.iloc[:, args.start_column:end_column])
                scaler_file = os.path.join(args.output_path, f"scaler_{os.path.basename(val_input_file)}.pkl")
                with open(scaler_file, "wb") as f:
                    pickle.dump(scaler, f)
            val_data_frames.append(val_df)
        val_data = pd.concat(val_data_frames, ignore_index=True)

        val_X = val_data.iloc[:, args.start_column : end_column].values
        val_y = val_data.iloc[:, args.y_column].values

        val_df_features = val_data.iloc[:, args.start_column : end_column]
        val_df_y = val_data.iloc[:, args.y_column]

    # Ensure same features are present in both input and validation data
    if args.val_input_path:
        # Ensure y columns are the same in input and validation data
        common_columns = set(df_features.columns)
        val_columns = set(val_df_features.columns)
        if common_columns == val_columns:
            val_df_features = val_df_features[df_features.columns]
        else:
            raise ValueError(
                "Column mismatch between input and validation data. Ordering fixed only if columns match exactly."
            )
    if args.val_input_path:
        if not df_y.name == val_df_y.name:
            raise ValueError("Target variable in input and validation data do not match. Check your data.")

    # If a correlation threshold is provided, filter features first
    if args.correlation_threshold is not None:
        # Compute Pearson correlations with target
        feature_corrs = df_features.apply(lambda col: pearsonr(col, df_y)[0])
        # Filter features
        selected_features = feature_corrs[feature_corrs.abs() >= args.correlation_threshold].index.tolist()
        if len(selected_features) == 0:
            raise ValueError("No features meet the correlation threshold. Lower the threshold or check your data.")
        print(
            f"Selected {len(selected_features)} features out of {len(original_metrics)} based on correlation threshold {args.correlation_threshold}."
        )
        print(f"Removed features: {set(original_metrics) - set(selected_features)}")
        df_features = df_features[selected_features]
        X = df_features.values
        metrics = selected_features

        # Filter validation data as well
        if args.val_input_path:
            val_df_features = val_df_features[selected_features]
            val_X = val_df_features.values
    else:
        metrics = original_metrics

    if args.standardize:
        scaler_file = os.path.join(args.output_path, "scaler.pkl")
        if not args.load_results and not args.scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            print("Standardized input data")
            # Save the StandardScaler object to the output directory
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            # Save the means and standard deviations to a JSON file
            scaler_stats = {
                "means": dict(zip(metrics, scaler.mean_.tolist())),
                "std_devs": dict(zip(metrics, scaler.scale_.tolist()))
            }
            scaler_stats_file = os.path.join(args.output_path, "scaler_stats.json")
            with open(scaler_stats_file, "w") as f:
                json.dump(scaler_stats, f, indent=4)
        else:
            if args.scaler:
                scaler_file = args.scaler
            # Load StandardScaler object from the output directory
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)

    if args.pca:
        pca_file = os.path.join(args.output_path, "pca.pkl")
        if not args.load_results:
            X, pca = perform_pca(X, args.pca)
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            metrics = [f"PC{i+1}" for i in range(args.pca)]
            # Save the PCA object to the output directory
            with open(pca_file, "wb") as f:
                pickle.dump(pca, f)
        else:
            # Load PCA object from the results dictionary
            with open(pca_file, "rb") as f:
                pca = pickle.load(f)
            X = pca.transform(X)

    # Determine methods to run
    if args.all_methods:
        methods_to_run = ["linear", "ridge", "lasso", "decision_tree", "random_forest", "xgboost", "svr"]
    else:
        methods_to_run = [args.method]

    for method in methods_to_run:
        print(f"\nRunning regression method: {method}")
        method_output_path = os.path.join(args.output_path, method)
        os.makedirs(method_output_path, exist_ok=True)

        if args.k_folds and args.k_folds > 1:
            print(f"Performing {args.k_folds}-fold cross-validation")
            kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
            pearson_corrs = []
            pearson_p_values = []
            spearman_corrs = []
            spearman_p_values = []
            fold_results = []
            fold_index = 0
            model_info = None
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train the model
                model = train_regression_model(X_train, y_train, method)
                # Compute R-squared on training set
                r_sq_train = model.score(X_train, y_train)
                # Get model coefficients or feature importances
                temp_model_info = get_model_coefficients(model, metrics)
                if not temp_model_info and method in ["decision_tree", "random_forest", "xgboost"]:
                    temp_model_info = get_feature_importances(model, metrics)

                # Evaluate on test set
                y_pred = model.predict(X_test)
                r_sq_test = r2_score(y_test, y_pred)
                mse_test = mean_squared_error(y_test, y_pred)
                # Compute correlations and save plot
                correlations = compute_correlations(y_pred, y_test, method_output_path, method, y_label, fold_index)
                pearson_corr = correlations["pearson_corr"]
                pearson_p_value = correlations["pearson_p_value"]
                spearman_corr = correlations["spearman_corr"]
                spearman_p_value = correlations["spearman_p_value"]
                # Plot residuals
                plot_residuals(y_pred, y_test, method_output_path, method, y_label, fold_index)
                # Save results for this fold
                fold_result = {
                    "fold": fold_index + 1,
                    "train_r_sq": r_sq_train,
                    "test_r_sq": r_sq_test,
                    "mse_test": mse_test,
                    "pearson_corr": pearson_corr,
                    "pearson_p_value": pearson_p_value,
                    "spearman_corr": spearman_corr,
                    "spearman_p_value": spearman_p_value,
                }
                fold_result.update(temp_model_info)
                fold_results.append(fold_result)
                pearson_corrs.append(pearson_corr)
                pearson_p_values.append(pearson_p_value)
                spearman_corrs.append(spearman_corr)
                spearman_p_values.append(spearman_p_value)
                fold_index += 1
                model_info = temp_model_info  # Keep updating to last fold info

            # Compute mean and standard deviation of correlations
            pearson_corr_mean = np.mean(pearson_corrs)
            pearson_corr_std = np.std(pearson_corrs)
            spearman_corr_mean = np.mean(spearman_corrs)
            spearman_corr_std = np.std(spearman_corrs)
            print(f"Mean Pearson correlation coefficient: {pearson_corr_mean}")
            print(f"Standard deviation of Pearson correlation coefficient: {pearson_corr_std}")
            print(f"Mean Spearman correlation coefficient: {spearman_corr_mean}")
            print(f"Standard deviation of Spearman correlation coefficient: {spearman_corr_std}")

            # Save overall results
            results = {
                "k_folds": args.k_folds,
                "pearson_corr_mean": pearson_corr_mean,
                "pearson_corr_std": pearson_corr_std,
                "spearman_corr_mean": spearman_corr_mean,
                "spearman_corr_std": spearman_corr_std,
                "folds": fold_results,
            }
            # Ensure JSON serializability
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    results[key] = float(value)
                elif isinstance(value, np.integer):
                    results[key] = int(value)
            results_file = os.path.join(method_output_path, "results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)

            # Combine variables if possible
            if model_info:
                # Determine importance dictionary
                if "coef" in model_info:
                    importance_dict = model_info["coef"]
                elif "feature_importances" in model_info:
                    importance_dict = model_info["feature_importances"]
                else:
                    importance_dict = {}

                if importance_dict:
                    df_combined = combine_variables_with_same_root(df_features, importance_dict)
                    # Add the target variable to df_combined
                    df_combined[y_label] = df_y.reset_index(drop=True)
                    # Plot and save the reduced correlation matrices and CSV
                    plot_full_correlation_matrix(
                        df_combined, method_output_path, method="pearson", filename_prefix="combined"
                    )
                    plot_full_correlation_matrix(
                        df_combined, method_output_path, method="spearman", filename_prefix="combined"
                    )
                else:
                    print(f"Cannot combine variables for method {method} due to lack of coefficients or importances.")
            else:
                print(f"Cannot combine variables for method {method} due to lack of coefficients or importances.")

            # If feature removal test is requested
            if args.feature_removal_test:
                print("\nPerforming feature removal test...")
                baseline_pearson = pearson_corr_mean
                baseline_spearman = spearman_corr_mean
                removal_results = []
                for feature_to_remove in metrics:
                    # Create a reduced feature set without this feature
                    reduced_features = [f for f in metrics if f != feature_to_remove]
                    # Get indexes of remaining features
                    indices = [metrics.index(f) for f in reduced_features]
                    X_reduced = X[:, indices]

                    # Run k-fold CV again with the reduced feature set
                    pearson_mean_removed, spearman_mean_removed = run_k_fold(
                        X_reduced, y, method, reduced_features, method_output_path, y_label, n_splits=args.k_folds
                    )

                    pearson_diff = pearson_mean_removed - baseline_pearson
                    spearman_diff = spearman_mean_removed - baseline_spearman

                    removal_results.append(
                        {
                            "removed_feature": feature_to_remove,
                            "pearson_corr_mean_removed": pearson_mean_removed,
                            "spearman_corr_mean_removed": spearman_mean_removed,
                            "pearson_corr_diff": pearson_diff,
                            "spearman_corr_diff": spearman_diff,
                        }
                    )

                # Save the removal results to CSV
                removal_df = pd.DataFrame(removal_results)
                removal_df.to_csv(os.path.join(method_output_path, "feature_removal_test_results.csv"), index=False)
                print("Feature removal test completed. Results saved to 'feature_removal_test_results.csv'.")

                # And for grouped:
                print("\nPerforming grouped feature removal test...")
                grouped_metrics = combine_variables_with_same_root(df_features, importance_dict)

                grouped_removal_results = []
                for feature_to_remove in grouped_metrics.columns:
                    print(f"Removing feature: {feature_to_remove}")
                    # Create a reduced feature set without this feature
                    reduced_features = [f for f in metrics if f[: f.find("_")] not in feature_to_remove or print(f)]
                    # Get indexes of remaining features
                    indices = [metrics.index(f) for f in reduced_features]
                    X_reduced = X[:, indices]

                    # Run k-fold CV again with the reduced feature set
                    pearson_mean_removed, spearman_mean_removed = run_k_fold(
                        X_reduced, y, method, reduced_features, method_output_path, y_label, n_splits=args.k_folds
                    )

                    pearson_diff = pearson_mean_removed - baseline_pearson
                    spearman_diff = spearman_mean_removed - baseline_spearman

                    grouped_removal_results.append(
                        {
                            "removed_feature": feature_to_remove,
                            "pearson_corr_mean_removed": pearson_mean_removed,
                            "spearman_corr_mean_removed": spearman_mean_removed,
                            "pearson_corr_diff": pearson_diff,
                            "spearman_corr_diff": spearman_diff,
                        }
                    )

                # Save the removal results to CSV
                grouped_removal_df = pd.DataFrame(grouped_removal_results)
                grouped_removal_df.to_csv(
                    os.path.join(method_output_path, "grouped_feature_removal_test_results.csv"), index=False
                )
                print(
                    "Grouped feature removal test completed. Results saved to 'grouped_feature_removal_test_results.csv'."
                )

        if args.sequential_test:
            print("\nPerforming sequential feature selection test...")

            # Forward selection
            sfs_forward = SequentialFeatureSelector(
                LinearRegression(), k_features=args.sequential_test, forward=True, scoring="r2", cv=10
            )
            sfs_forward.fit(X_with_names, y)
            selected_features_forward = list(sfs_forward.k_feature_names_)
            print(f"Selected features (forward): {selected_features_forward}")

            # Create a reduced feature set with only the selected features
            selected_features_forward = [metrics.index(f) for f in selected_features_forward]
            X_selected_forward = X_with_names.iloc[:, selected_features_forward].values

            # Run k-fold CV with the reduced feature set
            pearson_mean_selected_forward, spearman_mean_selected_forward = run_k_fold(
                X_selected_forward, y, method, selected_features_forward, method_output_path, y_label, n_splits=10
            )
            print(f"Pearson correlation with selected features (forward): {pearson_mean_selected_forward}")
            print(f"Spearman correlation with selected features (forward): {spearman_mean_selected_forward}")

            # Save forward selection results to CSV
            forward_results = {
                "selected_features": selected_features_forward,
                "pearson_corr_mean": pearson_mean_selected_forward,
                "spearman_corr_mean": spearman_mean_selected_forward,
            }
            forward_results_df = pd.DataFrame([forward_results])
            forward_results_df.to_csv(os.path.join(method_output_path, "forward_selection_results.csv"), index=False)

            # Backward selection
            sfs_backward = SequentialFeatureSelector(
                LinearRegression(), k_features=args.sequential_test, forward=False, scoring="r2", cv=10
            )
            sfs_backward.fit(X_with_names, y)
            selected_features_backward = list(sfs_backward.k_feature_names_)
            print(f"Selected features (backward): {selected_features_backward}")

            # Create a reduced feature set with only the selected features
            selected_indices_backward = [metrics.index(f) for f in selected_features_backward]
            X_selected_backward = X_with_names.iloc[:, selected_indices_backward].values

            # Run k-fold CV with the reduced feature set
            pearson_mean_selected_backward, spearman_mean_selected_backward = run_k_fold(
                X_selected_backward, y, method, selected_features_backward, method_output_path, y_label, n_splits=10
            )
            print(f"Pearson correlation with selected features (backward): {pearson_mean_selected_backward}")
            print(f"Spearman correlation with selected features (backward): {spearman_mean_selected_backward}")

            # Save backward selection results to CSV
            backward_results = {
                "selected_features": selected_features_backward,
                "pearson_corr_mean": pearson_mean_selected_backward,
                "spearman_corr_mean": spearman_mean_selected_backward,
            }
            backward_results_df = pd.DataFrame([backward_results])
            backward_results_df.to_csv(os.path.join(method_output_path, "backward_selection_results.csv"), index=False)

        else:
            # No k-fold or single split approach
            if args.shuffle_split:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
            elif args.val_input_path:
                X_train, X_test, y_train, y_test = X, val_X, y, val_y
            else:
                X_train, X_test, y_train, y_test = X, X, y, y

            if args.load_results:
                # Load the model from 'model.pkl'
                model_file = os.path.join(method_output_path, "model.pkl")
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_test)
                correlations = compute_correlations(y_pred, y_test, method_output_path, method, y_label)
                pearson_corr = correlations["pearson_corr"]
                pearson_p_value = correlations["pearson_p_value"]
                spearman_corr = correlations["spearman_corr"]
                spearman_p_value = correlations["spearman_p_value"]
                print(f"Pearson correlation coefficient: {pearson_corr}")
                print(f"Pearson p-value: {pearson_p_value}")
                print(f"Spearman correlation coefficient: {spearman_corr}")
                print(f"Spearman p-value: {spearman_p_value}")
                # Plot residuals
                plot_residuals(y_pred, y_test, method_output_path, method, y_label)
            else:
                # Train the model
                model = train_regression_model(X_train, y_train, method)
                # Compute R-squared on training set
                r_sq_train = model.score(X_train, y_train)
                print(f"R-squared on training set: {r_sq_train}")
                # Get model coefficients or feature importances
                model_info = get_model_coefficients(model, metrics)
                if not model_info and method in ["decision_tree", "random_forest", "xgboost"]:
                    model_info = get_feature_importances(model, metrics)

                # Save the model
                model_file = os.path.join(method_output_path, "model.pkl")
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)
                # Evaluate on test set
                y_pred = model.predict(X_test)
                r_sq_test = r2_score(y_test, y_pred)
                mse_test = mean_squared_error(y_test, y_pred)
                print(f"R-squared on test set: {r_sq_test}")
                print(f"Mean Squared Error on test set: {mse_test}")
                # Compute correlations and save plot
                correlations = compute_correlations(y_pred, y_test, method_output_path, method, y_label)
                pearson_corr = correlations["pearson_corr"]
                pearson_p_value = correlations["pearson_p_value"]
                spearman_corr = correlations["spearman_corr"]
                spearman_p_value = correlations["spearman_p_value"]
                print(f"Pearson correlation coefficient: {pearson_corr}")
                print(f"Pearson p-value: {pearson_p_value}")
                print(f"Spearman correlation coefficient: {spearman_corr}")
                print(f"Spearman p-value: {spearman_p_value}")
                # Plot residuals
                plot_residuals(y_pred, y_test, method_output_path, method, y_label)
                # Save results
                results = {
                    "train_r_sq": r_sq_train,
                    "test_r_sq": r_sq_test,
                    "mse_test": mse_test,
                    "pearson_corr": pearson_corr,
                    "pearson_p_value": pearson_p_value,
                    "spearman_corr": spearman_corr,
                    "spearman_p_value": spearman_p_value,
                }
                results.update(model_info if model_info else {})
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        results[key] = value.tolist()
                    elif isinstance(value, (np.float32, np.float64)):
                        results[key] = float(value)
                    elif isinstance(value, np.integer):
                        results[key] = int(value)
                results_file = os.path.join(method_output_path, "results.json")
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=4)

                # Optionally, print top features
                if method in ["linear", "ridge", "lasso", "decision_tree", "random_forest", "xgboost"] and model_info:
                    if "coef" in model_info:
                        importance_dict = model_info["coef"]
                        print("\nTop features based on model coefficients:")
                    elif "feature_importances" in model_info:
                        importance_dict = model_info["feature_importances"]
                        print("\nTop features based on feature importances:")
                    else:
                        importance_dict = {}
                        print("\nNo coefficients or feature importances available.")

                    if importance_dict:
                        sorted_features = sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
                        for feature, importance in sorted_features[:10]:
                            print(f"{feature}: {importance}")

                        # Combine variables with the same root name using the coefficients/importances
                        df_combined = combine_variables_with_same_root(df_features, importance_dict)
                        # Add the target variable to df_combined
                        df_combined[y_label] = df_y.reset_index(drop=True)
                        # Plot and save the reduced correlation matrices and CSV
                        plot_full_correlation_matrix(
                            df_combined, method_output_path, method="pearson", filename_prefix="combined"
                        )
                        plot_full_correlation_matrix(
                            df_combined, method_output_path, method="spearman", filename_prefix="combined"
                        )
                    else:
                        print(
                            f"Cannot combine variables for method {method} due to lack of coefficients or importances."
                        )
                else:
                    print(f"Cannot combine variables for method {method} due to lack of coefficients or importances.")

    # Plot Pearson correlation heatmap between features and target variable
    plot_feature_correlations(df_features, df_y, args.output_path, y_label, method="pearson")

    # Plot Spearman correlation heatmap between features and target variable
    plot_feature_correlations(df_features, df_y, args.output_path, y_label, method="spearman")

    # Combine features and target variable into one DataFrame for original correlation matrices
    df_full = df_features.copy()
    df_full[y_label] = df_y

    # Plot full Pearson correlation matrix for original data
    plot_full_correlation_matrix(df_full, args.output_path, method="pearson", filename_prefix="full")

    # Plot full Spearman correlation matrix for original data
    plot_full_correlation_matrix(df_full, args.output_path, method="spearman", filename_prefix="full")

    # Plot metrics against the target variable
    plot_metrics_vs_y(df_full, y_label, args.output_path)


if __name__ == "__main__":
    main()
