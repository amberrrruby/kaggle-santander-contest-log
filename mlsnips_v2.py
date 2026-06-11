'''
ML utility snippets for common EDA, data cleaning, feature analysis, and time-series tasks.
Organized by functionality with consistent naming conventions.
'''
# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA SUMMARY & INSPECTION (Numerical)
# ============================================================================

def eda_inspect_df(df: pd.DataFrame) -> pd.DataFrame:
    """Quick inspect of a DataFrame: shape, info, dtypes, nulls, basic stats.

    Prints comprehensive summary and returns ``df.describe()`` (numeric only).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.

    Returns
    -------
    pd.DataFrame
        The result of ``df.describe()`` (numeric summary).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    print(f">>> Shape: {df.shape}")
    print("=" * 50)
    print(">>> Info:")
    df.info()
    print("=" * 50)
    print(">>> Summary datatypes:")
    print(df.dtypes.unique())
    print("=" * 50)
    nulls = df.isna().sum().sort_values(ascending=False)
    print(">>> Nulls (top 20):")
    print(nulls.head(20))
    print("=" * 50)
    print(">>> Basic stats (numeric columns):")
    return df.describe()


def get_grouped_colnames(df: pd.DataFrame) -> dict:
    """Group column names by datatype: numeric, categorical, datetime, boolean, other."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    boolean = df.select_dtypes(include=['bool']).columns.tolist()
    other = [c for c in df.columns if c not in (numeric + categorical + datetime + boolean)]
    return {
        'numeric': numeric,
        'categorical': categorical,
        'datetime': datetime,
        'boolean': boolean,
        'other': other,
    }


def summarize_numeric(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Extended numeric summary: count, missing, mean, std, min, q25, median, q75, max, skew, kurtosis, nunique."""
    num = df.select_dtypes(include=[np.number]) if cols is None else df[cols]
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc = desc.rename(columns={'25%': 'q25', '50%': 'median', '75%': 'q75'})
    desc['missing'] = num.isna().sum()
    desc['nunique'] = num.nunique()
    desc['skew'] = num.skew()
    desc['kurtosis'] = num.kurtosis()
    cols_order = ['count', 'missing', 'nunique', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'skew', 'kurtosis']
    return desc[cols_order]


def summarize_dataframe(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Compact dataframe summary: n_rows, n_cols, memory, missing, duplicates, head/tail, numeric summary."""
    info = {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
        'n_missing_columns': int((df.isna().sum() > 0).sum()),
        'total_missing': int(df.isna().sum().sum()),
        'n_duplicate_rows': int(df.duplicated().sum()),
        'head': df.head(top_n),
        'tail': df.tail(top_n),
        'numeric_summary': summarize_numeric(df)
    }
    return info


def eda_catcounts(df: pd.DataFrame, include_na: bool = False) -> pd.DataFrame:
    """Value counts for all categorical columns (consolidated table).

    Returns
    -------
    pd.DataFrame
        Columns: ['column', 'category', 'count']
    """
    records = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        vc = df[col].value_counts(dropna=not include_na)
        for category, count in vc.items():
            records.append({'column': col, 'category': category, 'count': count})
    return pd.DataFrame(records)


def missing_value_report(df: pd.DataFrame, show_bar: bool = True, show_heatmap: bool = False) -> pd.DataFrame:
    """Missing values per column with optional visualizations."""
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    out = pd.DataFrame({'missing_count': miss, 'percentage_missing': miss / len(df)})

    if show_bar and not out.empty:
        ax = out.head(20).missing_count.plot(kind='bar', figsize=(10, 4), title='Top missing columns')
        ax.set_ylabel('missing_count')
        plt.tight_layout()
        plt.show()

    if show_heatmap and not out.empty:
        try:
            subset = df[out.index].isnull().astype(int)
            plt.figure(figsize=(12, 3))
            sns.heatmap(subset.T, cmap='viridis', cbar=False)
            plt.yticks(rotation=0)
            plt.title('Missingness heatmap')
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    return out


# ============================================================================
# 2. PLOTTING - NUMERIC COLUMNS
# ============================================================================

def plot_numeric_hist(df: pd.DataFrame, bins: int = 25, figsize=(16, 20)):
    """Histograms for all numeric columns."""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        raise ValueError("No numeric columns found to plot")
    return num.hist(bins=bins, figsize=figsize)


def plot_numeric_distribution(df: pd.DataFrame, col: str, log1p_transform: bool = False, figsize=(12, 8)):
    """2x2 grid: histogram+kde, CDF, box plot, violin plot for a single numeric column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    data = df[col].dropna()
    if log1p_transform:
        data = np.log1p(data)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    sns.histplot(data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram with KDE")

    sns.ecdfplot(data, ax=axes[1, 0])
    axes[1, 0].set_title("CDF")

    sns.boxplot(x=data, ax=axes[0, 1])
    axes[0, 1].set_title("Box Plot")

    sns.violinplot(x=data, ax=axes[1, 1])
    axes[1, 1].set_title("Violin Plot")

    plt.tight_layout()
    plt.show()


def plot_numeric_qq_and_dist(series: pd.Series, transform: str = None, figsize=(12, 5)):
    """Side-by-side QQ-plot and distribution for a numeric series. Transform can be 'log1p' or 'boxcox'."""
    ser = series.dropna()
    if transform == 'log1p':
        ser = np.log1p(ser)
    elif transform == 'boxcox':
        from scipy import stats
        if (ser <= 0).any():
            raise ValueError('boxcox requires all positive values')
        ser, _ = stats.boxcox(ser)

    import statsmodels.api as sm
    from scipy.stats import norm
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sm.qqplot(ser, line='45', ax=ax[0])
    sns.histplot(ser, kde=True, stat='density', ax=ax[1])
    x = np.linspace(ser.min(), ser.max(), 100)
    ax[1].plot(x, norm.pdf(x, loc=ser.mean(), scale=ser.std()), color='red')
    ax[0].set_title('QQ-plot')
    ax[1].set_title('Distribution (with normal fit)')
    plt.tight_layout()
    plt.show()


def plot_numeric_corr(df: pd.DataFrame, method: str = 'pearson', annotate: bool = True, 
                      top_k: int = None, figsize=(10, 8)):
    """Correlation matrix heatmap. If ``top_k``, plot only top-k correlated features."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError('Need at least two numeric columns for correlation matrix')
    corr = num.corr(method=method)
    if top_k is not None and top_k < corr.shape[0]:
        ranks = corr.abs().max().sort_values(ascending=False).head(top_k).index.tolist()
        corr = corr.loc[ranks, ranks]
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annotate, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_numeric_pairplot(df: pd.DataFrame, hue_col: str = None, vars: list = None):
    """Pairplot for numeric variables (optionally with a hue)."""
    if vars is None:
        vars = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_plot = vars if hue_col is None else (vars + [hue_col] if hue_col not in vars else vars)
    sns.pairplot(df[cols_to_plot], hue=hue_col, corner=True)
    plt.show()


def plot_numeric_scatter(df: pd.DataFrame, x_col: str, y_col: str, alpha: float = 0.2, 
                        log_x: bool = False, figsize=(8, 6)):
    """Scatter plot for two numeric columns."""
    plt.figure(figsize=figsize)
    x_data = np.log1p(df[x_col]) if log_x else df[x_col]
    sns.scatterplot(x=x_data, y=df[y_col], alpha=alpha, color='steelblue')
    plt.xlabel(f"{x_col}{'(log scale)' if log_x else ''}")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()


def plot_numeric_scatter_hue(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, figsize=(10, 6)):
    """Scatter plot for two numeric columns colored by a third column."""
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='husl')
    plt.tight_layout()
    plt.show()


def plot_numeric_scatter_regr(df: pd.DataFrame, x_col: str, y_col: str, figsize=(8, 6)):
    """Scatter plot with regression line for two numeric columns."""
    plt.figure(figsize=figsize)
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha': 0.3})
    plt.tight_layout()
    plt.show()


def plot_numeric_joint(df: pd.DataFrame, x_col: str, y_col: str, kind: str = 'scatter'):
    """Joint plot (scatter + marginal distributions) for two numeric columns. Kind: 'scatter', 'hex', 'kde'."""
    sns.jointplot(x=x_col, y=y_col, data=df, kind=kind)
    plt.show()


# ============================================================================
# 3. PLOTTING - CATEGORICAL COLUMNS
# ============================================================================

def plot_cat_counts(df: pd.DataFrame, col: str, incl_percentage: bool = True, figsize=(10, 5)):
    """Countplot for a categorical column with optional percentage labels."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    values = df[col].value_counts(dropna=False)
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=col, data=df, order=values.index)

    if incl_percentage:
        total = len(df)
        ymax = values.max()
        for i, v in enumerate(values.values):
            pct = 100 * v / total
            ax.text(i, v + ymax * 0.02, f"{pct:0.2f}%", ha='center', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cat_numeric_box(df: pd.DataFrame, cat_col: str, num_col: str, figsize=(10, 5)):
    """Box plot showing numeric distribution grouped by a categorical column."""
    plt.figure(figsize=figsize)
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cat_numeric_violin(df: pd.DataFrame, cat_col: str, num_col: str, figsize=(10, 5)):
    """Violin plot showing numeric distribution grouped by a categorical column."""
    plt.figure(figsize=figsize)
    sns.violinplot(x=cat_col, y=num_col, data=df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. PLOTTING - MISSING DATA & CORRELATIONS
# ============================================================================

def plot_missing_matrix(df: pd.DataFrame, figsize=(12, 3), max_cols: int = 100):
    """Heatmap showing missingness patterns (columns x rows)."""
    subset_cols = df.columns[:max_cols]
    subset = df[subset_cols].isnull().astype(int)
    plt.figure(figsize=figsize)
    sns.heatmap(subset.T, cmap='binary', cbar=False)
    plt.title(f'Missingness (columns x rows, max {max_cols} cols)')
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. FEATURE ANALYSIS
# ============================================================================

def corr_with_target(df: pd.DataFrame, target: str, top_n: int = 20) -> pd.DataFrame:
    """Top N numeric features correlated (by absolute value) with target."""
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise ValueError('target not numeric or not in dataframe')
    corrs = numeric.corr()[target].drop(target).abs().sort_values(ascending=False)
    return pd.DataFrame({'feature': corrs.index, 'abs_corr': corrs.values}).head(top_n)


def pca_cumvar_plot(df: pd.DataFrame, threshold: float = 0.9, figsize=(12, 8)):
    """PCA cumulative variance plot with threshold line."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    X_num = df.select_dtypes(include=[np.number])
    if X_num.empty:
        raise ValueError("No numeric columns found")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    pca = PCA()
    pca.fit(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f'Components needed to reach {int(threshold * 100)}% variance: {n_components}')

    plt.figure(figsize=figsize)
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', lw=2)
    plt.axhline(y=threshold, linestyle='--', color='red', label=f'{int(threshold*100)}% threshold')
    plt.xlabel('# Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. TIME SERIES ANALYSIS
# ============================================================================

def plot_ts_series(series: pd.Series, figsize=(14, 5), title: str = None, ylabel: str = 'Value'):
    """Plot a time series."""
    plt.figure(figsize=figsize)
    plt.plot(series.index, series.values, linewidth=1.5, color='steelblue')
    plt.title(title or 'Time Series')
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ts_decompose(series: pd.Series, period: int = None, figsize=(14, 10)):
    """Seasonal decomposition: trend, seasonal, residual."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if period is None:
        period = 52 if len(series) > 104 else 12  # Default: annual for daily data, monthly for higher freq
    
    decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate='fill')
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    series.plot(ax=axes[0], color='steelblue')
    axes[0].set_ylabel('Original')
    axes[0].grid(True, alpha=0.3)
    
    decomposition.trend.plot(ax=axes[1], color='orange')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    decomposition.seasonal.plot(ax=axes[2], color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    decomposition.resid.plot(ax=axes[3], color='red')
    axes[3].set_ylabel('Residual')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_ts_acf_pacf(series: pd.Series, lags: int = 40, figsize=(14, 5)):
    """ACF and PACF plots."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation (ACF)')
    
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title('Partial Autocorrelation (PACF)')
    
    plt.tight_layout()
    plt.show()


def plot_ts_rolling(series: pd.Series, windows: list = None, figsize=(14, 6)):
    """Plot series with rolling mean and standard deviation."""
    if windows is None:
        windows = [7, 30]
    
    plt.figure(figsize=figsize)
    plt.plot(series.index, series.values, label='Original', linewidth=1, alpha=0.7)
    
    for w in windows:
        rolling_mean = series.rolling(window=w).mean()
        rolling_std = series.rolling(window=w).std()
        plt.plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean (w={w})', linewidth=1.5)
        plt.fill_between(rolling_std.index, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std, 
                         alpha=0.1, label=f'±2σ (w={w})')
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ts_lag(series: pd.Series, lags: int = 9, figsize=(12, 10)):
    """Lag scatter plots."""
    from pandas.plotting import lag_plot
    
    fig, axes = plt.subplots((lags + 2) // 3, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(1, lags + 1):
        lag_plot(series, lag=i, ax=axes[i-1], alpha=0.5)
        axes[i-1].set_title(f'Lag {i}')
    
    for j in range(lags, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def ts_stationarity_test(series: pd.Series, name: str = 'Series', verbose: bool = True) -> dict:
    """Augmented Dickey-Fuller (ADF) test for stationarity."""
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series.dropna(), autolag='AIC')
    
    output = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'is_stationary': result[1] < 0.05  # Critical: p < 0.05
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ADF Test Results for {name}")
        print(f"{'='*60}")
        print(f"ADF Statistic:    {result[0]:.6f}")
        print(f"P-value:          {result[1]:.6f}")
        print(f"# Lags Used:      {result[2]}")
        print(f"# Observations:   {result[3]}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")
        print(f"\nStationary: {'YES (p < 0.05)' if output['is_stationary'] else 'NO (p >= 0.05)'}")
        print(f"{'='*60}\n")
    
    return output


def ts_rolling_stats(series: pd.Series, window: int = 7) -> pd.DataFrame:
    """Rolling statistics for a time series."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()
    
    return pd.DataFrame({
        'value': series,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'rolling_min': rolling_min,
        'rolling_max': rolling_max
    })


# ============================================================================
# 7. DATA PREPROCESSING & IMPUTATION
# ============================================================================

def impute_categorical_fill(df: pd.DataFrame, cols: list, fill_value: str = 'None', tokens: list = None) -> pd.DataFrame:
    """Fill categorical columns, first normalizing common NA-like tokens."""
    df = df.copy()
    if tokens is None:
        tokens = ['NA', 'None', 'nan', '']

    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df[c].replace(tokens, np.nan)
        df[c] = df[c].fillna(fill_value)
    return df


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute numeric column using median per group."""
    def __init__(self, column: str, groupby_col: str, fill_value=None):
        self.column = column
        self.groupby_col = groupby_col
        self.fill_value = fill_value

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.medians_ = df.groupby(self.groupby_col)[self.column].median().to_dict()
        self.global_median_ = df[self.column].median() if self.fill_value is None else self.fill_value
        return self

    def transform(self, X):
        df = X.copy()
        if self.column not in df.columns or self.groupby_col not in df.columns:
            return df

        def _impute(row):
            val = row[self.column]
            if pd.notna(val):
                return val
            grp = row[self.groupby_col]
            return self.medians_.get(grp, self.global_median_)

        df[self.column] = df.apply(_impute, axis=1)
        return df


def encode_ordinal(series: pd.Series, mapping: dict, na_value=np.nan) -> pd.Series:
    """Map ordinal strings to integers (unknowns → NaN or na_value)."""
    return series.map(mapping).fillna(na_value)


def cap_outliers(df: pd.DataFrame, cols: list, method: str = 'iqr', 
                 lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    """Cap outliers using IQR or quantile method."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        ser = df[c]
        if method == 'iqr':
            q1 = ser.quantile(0.25)
            q3 = ser.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
        else:
            lo = ser.quantile(lower_quantile)
            hi = ser.quantile(upper_quantile)
        df[c] = ser.clip(lower=lo, upper=hi)
    return df


# ============================================================================
# 8. MODEL TRAINING & CROSS-VALIDATION
# ============================================================================

def make_cv_splits(y, n_splits: int = 5, bins: int = 10, random_state: int = 42):
    """Stratified CV splits by binning target (useful for regression)."""
    y = pd.Series(y).reset_index(drop=True)
    try:
        y_binned = pd.qcut(y, q=min(bins, len(y.unique())), duplicates='drop')
    except Exception:
        y_binned = pd.cut(y, bins=bins)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X=np.zeros(len(y)), y=y_binned.astype(str)))


def train_and_report_trees(X_train, X_val, y_train, y_val, report_models: bool = True) -> pd.DataFrame:
    """Train tree-based regression models and report metrics (MAE, RMSE, R²)."""
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    import time

    tree_models = {
        "random-forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "extra-trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient-boosting": GradientBoostingRegressor(random_state=42),
        "xgboost": XGBRegressor(tree_method="hist", random_state=42, n_estimators=300, 
                               learning_rate=0.1, verbosity=0),
        "light-gbm": LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.1),
    }

    if report_models:
        print("Training tree models:")
        for name in tree_models:
            print(f" - {name}")

    records = []
    for name, model in tree_models.items():
        start_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        start_pred = time.time()
        preds = model.predict(X_val)
        pred_time = time.time() - start_pred

        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        records.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time
        })

    return pd.DataFrame(records).sort_values("mae")
