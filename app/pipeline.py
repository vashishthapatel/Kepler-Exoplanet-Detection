import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import optuna

# --- Preprocessing ---
def preprocess_flux(df, sigma=10):
    vals = df.values.astype(np.float64)
    mean = vals.mean(axis=1, keepdims=True)
    std  = vals.std(axis=1, keepdims=True)
    std[std == 0] = 1
    vals = (vals - mean) / std
    vals = gaussian_filter1d(vals, sigma=sigma, axis=1)
    return vals

# --- Feature Engineering ---
def extract_features(flux_matrix):
    all_features = []
    for i in range(flux_matrix.shape[0]):
        flux = flux_matrix[i]
        f = {}
        f['std']      = np.std(flux)
        f['skewness'] = scipy_stats.skew(flux)
        f['kurtosis'] = scipy_stats.kurtosis(flux)
        f['range']    = np.max(flux) - np.min(flux)
        f['min']      = np.min(flux)
        f['max']      = np.max(flux)
        f['median']   = np.median(flux)
        f['mean_abs_dev'] = np.mean(np.abs(flux - np.mean(flux)))
        f['iqr']      = np.percentile(flux, 75) - np.percentile(flux, 25)
        mean_f = np.mean(flux)
        std_f  = np.std(flux)
        for k in [2, 3]:
            threshold = mean_f - k * std_f
            below = flux < threshold
            n_below = np.sum(below)
            f[f'n_dips_{k}sigma']        = n_below
            f[f'frac_dips_{k}sigma']     = n_below / len(flux)
            dip_vals = flux[below]
            f[f'mean_dip_depth_{k}sigma'] = np.mean(dip_vals) if n_below > 0 else 0
            f[f'max_dip_depth_{k}sigma']  = np.min(dip_vals) if n_below > 0 else 0
            diffs = np.diff(below.astype(int))
            n_regions = np.sum(diffs == 1)
            f[f'n_dip_regions_{k}sigma'] = n_regions
        p5  = np.percentile(flux, 5)
        p95 = np.percentile(flux, 95)
        p50 = np.percentile(flux, 50)
        f['p5_to_median']  = p5 / p50 if p50 != 0 else 0
        f['p95_minus_p5']  = p95 - p5
        f['p1']  = np.percentile(flux, 1)
        f['p99'] = np.percentile(flux, 99)
        fft_vals = np.abs(np.fft.rfft(flux))[1:]
        fft_power = fft_vals ** 2
        sorted_power = np.sort(fft_power)[::-1]
        for j in range(min(10, len(sorted_power))):
            f[f'fft_power_top_{j}'] = sorted_power[j]
        f['fft_power_mean']  = np.mean(fft_power)
        f['fft_power_std']   = np.std(fft_power)
        f['fft_power_max']   = np.max(fft_power)
        f['fft_peak_ratio'] = np.max(fft_power) / np.mean(fft_power) if np.mean(fft_power) > 0 else 0
        flux_centered = flux - np.mean(flux)
        acf_full = np.correlate(flux_centered, flux_centered, mode='full')
        acf = acf_full[len(acf_full)//2:]
        acf = acf / acf[0] if acf[0] != 0 else acf
        acf_no_zero = acf[1:]
        f['acf_max']     = np.max(acf_no_zero) if len(acf_no_zero) > 0 else 0
        f['acf_max_lag'] = np.argmax(acf_no_zero) + 1 if len(acf_no_zero) > 0 else 0
        for lag in [50, 100, 200, 500]:
            f[f'acf_lag_{lag}'] = acf[lag] if lag < len(acf) else 0
        window = 200
        if len(flux) >= window:
            n_windows = len(flux) - window + 1
            rolling_std = np.array([np.std(flux[j:j+window]) for j in range(0, n_windows, window//4)])
            f['rolling_std_max']   = np.max(rolling_std)
            f['rolling_std_min']   = np.min(rolling_std)
            f['rolling_std_range'] = np.max(rolling_std) - np.min(rolling_std)
            f['rolling_std_ratio'] = np.max(rolling_std) / np.min(rolling_std) if np.min(rolling_std) > 0 else 0
        else:
            f['rolling_std_max']   = 0
            f['rolling_std_min']   = 0
            f['rolling_std_range'] = 0
            f['rolling_std_ratio'] = 0
        all_features.append(f)
    return pd.DataFrame(all_features)
