import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Force high resolution ----------------
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# -------------------------------------------------------

# ----------------------- User inputs -----------------------
FILE_PATH = r"C:\Users\Aziz\Desktop\DesktopOctober2024\ExamenDoctoralOctober2024\PhDExamArchives\Article-1\correction\SSC_Flow_Trends_1979-2024\DailySSCDuration_1979-2024.xlsx"

y_label            = 'SSC (mg/L)'  
title_basic        = 'Daily SSC Duration Curves (1979–2024)'
title_sw           = 'Daily SSC Duration Curves: Summer vs Winter (1979–2024)'
title_month_prefix = 'Monthly SSC Duration Curves (1979–2024) - '
summer_months      = [6, 7, 8, 9]  # Jun–Sep
# -----------------------------------------------------------


# ----------------------- Helpers ---------------------------
def robust_parse_date_series(s: pd.Series) -> pd.Series:
    try:
        if pd.api.types.is_numeric_dtype(s):
            base = pd.Timestamp('1899-12-30')  
            dt = pd.to_datetime(s, unit='D', origin=base)
            if not dt.isna().all():
                return dt
    except Exception:
        pass
    dt = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    if dt.isna().all():
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'):
            dt = pd.to_datetime(s, format=fmt, errors='coerce')
            if not dt.isna().all():
                break
    if dt.isna().any():
        raise ValueError("Could not parse the Date column.")
    return dt


def compute_exceedance_curve(values, sort_direction='descend'):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    if sort_direction.lower().startswith('desc'):
        v_sorted = np.sort(v)[::-1]
    else:
        v_sorted = np.sort(v)
    n = v_sorted.size
    probs = (np.arange(1, n + 1) / (n + 1.0)) * 100.0
    return probs, v_sorted


def duration_plot(
    Y,
    Legend=None,
    y_label='',
    x_label='Frequency of exceeding (%)',
    Title='',
    SortDirection='descend',
    LineWidth=2,
    ProbabilityOnYAxis=False,
    SummerMonthsPartition=None,  
    MonthPartition=None          
):
    fig, ax = plt.subplots(figsize=(8, 6))

    def plot_series(y, label):
        px, py = compute_exceedance_curve(y, SortDirection)
        if px.size == 0:
            return
        if ProbabilityOnYAxis:
            ax.plot(py, px, linewidth=LineWidth, label=label)
        else:
            ax.plot(px, py, linewidth=LineWidth, label=label)

    if SummerMonthsPartition is None and MonthPartition is None:
        arr = np.asarray(Y)
        if arr.ndim == 1:
            arr = arr[:, None]
        for i in range(arr.shape[1]):
            lbl = Legend[i] if Legend and i < len(Legend) else f'S{i+1}'
            plot_series(arr[:, i], lbl)

    elif SummerMonthsPartition is not None and MonthPartition is None:
        date_series, summer_months_local = SummerMonthsPartition
        date_series = pd.to_datetime(date_series)
        is_summer = pd.Series(date_series).dt.month.isin(summer_months_local).to_numpy()

        arr = np.asarray(Y)
        if arr.ndim == 1:
            arr = arr[:, None]

        for i in range(arr.shape[1]):
            lbl_base = Legend[i] if Legend and i < len(Legend) else f'S{i+1}'
            plot_series(arr[is_summer, i], f'{lbl_base} (Summer)')
            plot_series(arr[~is_summer, i], f'{lbl_base} (Winter)')

    elif MonthPartition is not None:
        date_series = pd.to_datetime(MonthPartition)
        y = np.asarray(Y).squeeze()
        if y.ndim != 1:
            raise ValueError('MonthPartition mode expects a single series (1D).')

        for m in range(1, 13):
            mask = (pd.Series(date_series).dt.month == m).to_numpy()
            if not mask.any():
                continue
            label = datetime(2000, m, 1).strftime('%b')
            plot_series(y[mask], label)

    ax.set_title(Title)
    if ProbabilityOnYAxis:
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle=':')
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_box_aspect(0.7)

    # ---------------- Logarithmic X-axis ----------------
    ax.set_xscale('log')
    ax.set_xlim(left=0.1, right=100)  # avoid log(0)
    # ----------------------------------------------------

    plt.show()


# ------------------- Read & prepare data -------------------
T = pd.read_excel(FILE_PATH, header=0)
DateCol = T.iloc[:, 0]
Date = robust_parse_date_series(DateCol)

Y = T.iloc[:, 1:].to_numpy(dtype=float)
river_names = list(T.columns[1:])

if Y.shape[1] != 6:
    raise ValueError(f'Expected 6 river columns (2..7). Found {Y.shape[1]}.')

startD = pd.Timestamp(1979, 1, 1)
endD   = pd.Timestamp(2024, 12, 31)
in_range = (Date >= startD) & (Date <= endD)
Date = Date[in_range].reset_index(drop=True)
Y = Y[in_range.values, :]

mask_all = ~np.isnan(Y).any(axis=1)
if not mask_all.all():
    print(f'Note: removing {int((~mask_all).sum())} rows with any NaN across rivers.')
Date = Date[mask_all].reset_index(drop=True)
Y = Y[mask_all, :]

print(f'Rows kept: {len(Date)} (from {Date.min().date()} to {Date.max().date()})')


# ---------- 1) Basic curve ----------
duration_plot(
    Y,
    Legend=river_names,
    y_label=y_label,
    x_label='Frequency of exceeding (%)',
    Title=title_basic
)

# ---------- 2) Summer & Winter overlay ----------
duration_plot(
    Y,
    Legend=river_names,
    y_label=y_label,
    x_label='Frequency of exceeding (%)',
    Title=title_sw,
    SummerMonthsPartition=(Date, summer_months)
)

# ---------- 2b) Summer-only ----------
idx_summer = Date.dt.month.isin(summer_months).to_numpy()
Y_sum = Y[idx_summer, :]
mask_sum = ~np.isnan(Y_sum).any(axis=1)
Y_sum = Y_sum[mask_sum, :]

duration_plot(
    Y_sum,
    Legend=river_names,
    y_label=y_label,
    x_label='Frequency of exceeding (%)',
    Title='Daily SSC Duration Curves: Summer Only (1979–2024)'
)

# ---------- 2c) Winter-only ----------
idx_winter = ~idx_summer
Y_win = Y[idx_winter, :]
mask_win = ~np.isnan(Y_win).any(axis=1)
Y_win = Y_win[mask_win, :]

duration_plot(
    Y_win,
    Legend=river_names,
    y_label=y_label,
    x_label='Frequency of exceeding (%)',
    Title='Daily SSC Duration Curves: Winter Only (1979–2024)'
)

# ---------- 3) Monthly traces for each river ----------
for r in range(6):
    y_river = Y[:, r]
    river   = river_names[r]
    duration_plot(
        y_river,
        Legend=[river],
        y_label=y_label,
        x_label='Frequency of exceeding (%)',
        Title=f"{title_month_prefix}{river.replace('_', ' ')}",
        MonthPartition=Date
    )
