import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np
import glob
import os

TARGET_TTFT=5000 # ms

def get_smart_interval(data_range, target_ticks=10):
    """
    Calculate smart tick interval based on data range

    Args:
        data_range: The range of data (max - min)
        target_ticks: Target number of ticks (default: 10)

    Returns:
        A nice round number for tick interval
    """
    if data_range == 0:
        return 1

    # Calculate raw interval
    raw_interval = data_range / target_ticks

    # Round to nice numbers: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000...
    magnitude = 10 ** np.floor(np.log10(raw_interval))
    residual = raw_interval / magnitude

    if residual <= 1.5:
        nice_interval = 1 * magnitude
    elif residual <= 3:
        nice_interval = 2 * magnitude
    elif residual <= 7:
        nice_interval = 5 * magnitude
    else:
        nice_interval = 10 * magnitude

    return nice_interval

def find_all_result_files(case):
    """Find all benchmark result files for a given case/scenario.

    Supports both old format (benchmark_results_{case}.csv) and
    new format (benchmark_results_{case}_{timestamp}.csv)

    Args:
        case: The scenario name (e.g., 'round_robin', 'zipfian', etc.)

    Returns:
        List of (file_path, output_name) tuples, sorted by modification time
    """
    result_files = []

    # Find new format files (with timestamp)
    pattern = f"benchmark_results_{case}_*.csv"
    matching_files = glob.glob(pattern)

    for file_path in matching_files:
        # Extract base name without .csv extension for output
        output_name = os.path.splitext(file_path)[0]
        result_files.append((file_path, output_name))

    # Check for old format file (without timestamp)
    old_format_file = f"benchmark_results_{case}.csv"
    if os.path.exists(old_format_file):
        output_name = f"benchmark_results_{case}"
        result_files.append((old_format_file, output_name))

    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)

    return result_files

# Read data
cases = ["round_robin", "zipfian", "bursty", "rag"]
file_data_list = []  # List of (case, file_path, output_name, df)
all_models = set()

# Step 1: Find and read all result files
for case in cases:
    result_files = find_all_result_files(case)

    if not result_files:
        print(f"Warning: No result file found for case '{case}'")
        continue

    for file_path, output_name in result_files:
        try:
            print(f"Loading: {file_path}")
            df = pd.read_csv(file_path)
            # Preprocessing: Normalize start time to start from 0
            df['relative_start_time'] = df['start_time'] - df['start_time'].min()
            file_data_list.append((case, file_path, output_name, df))
            all_models.update(df['model'].unique())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Set unified colors and order
unique_models = sorted(list(all_models))
palette = dict(zip(unique_models, sns.color_palette("tab10", len(unique_models))))

# Step 2: Generate plots for each file
for case, file_path, output_name, df in file_data_list:
    print(f"\nGenerating plot for: {file_path}")

    # Set up canvas
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. System Stability (Time vs TTFT)
    sns.scatterplot(data=df, x='relative_start_time', y='ttft_ms', hue='model', palette=palette, hue_order=unique_models, ax=axes[0, 0])
    axes[0, 0].set_title('1. System Stability: TTFT over Time')
    axes[0, 0].set_xlabel('Time Elapsed (s)')

    # Auto-adjust X-axis (time) interval
    time_range = df['relative_start_time'].max() - df['relative_start_time'].min()
    time_interval = get_smart_interval(time_range, target_ticks=10)
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(time_interval))

    # Auto-adjust Y-axis (TTFT) interval
    ttft_range = df['ttft_ms'].max() - df['ttft_ms'].min()
    ttft_interval = get_smart_interval(ttft_range, target_ticks=10)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(ttft_interval))
    axes[0, 0].grid(True)

    # 2. Distribution Detection (TTFT Distribution)
    sns.histplot(data=df, x='ttft_ms', kde=True, hue='model', palette=palette, hue_order=unique_models, element="step", ax=axes[0, 1])
    
    # Add Total line (Black)
    sns.kdeplot(data=df, x='ttft_ms', color='black', linewidth=2, ax=axes[0, 1], label='Total KDE')
    # Note: kdeplot is a density plot, histplot defaults to count. If displayed on the same axis, there may be scaling issues.
    # But the user requested a "total line", usually referring to the overall trend.
    # To make the Total line easier to compare with the Histogram, we can use histplot(element="poly", fill=False) to draw the Total outline
    # Or we can use dual axes, but that would make the chart complicated.
    # For simplicity, we draw an unfilled histplot to represent Total
    sns.histplot(data=df, x='ttft_ms', kde=True, color='red', element="step", fill=False, ax=axes[0, 1], label='Total')

    # Add TTFT 5000ms line
    axes[0, 1].axvline(TARGET_TTFT, color='red', linestyle='--', linewidth=2, label=f'Target {TARGET_TTFT}ms')
    
    # Calculate and display percentage
    pct_within = (df['ttft_ms'] <= TARGET_TTFT).mean() * 100
    # Get Y-axis range to determine text position
    y_min, y_max = axes[0, 1].get_ylim()
    axes[0, 1].text(TARGET_TTFT + 100, y_max * 0.9, f'{pct_within:.1f}% <= {TARGET_TTFT}ms', color='red', fontweight='bold')

    axes[0, 1].set_title('2. Latency Distribution (Check for Bimodal)')

    # Auto-adjust X-axis (TTFT distribution) interval
    ttft_dist_interval = get_smart_interval(ttft_range, target_ticks=10)
    axes[0, 1].xaxis.set_major_locator(MultipleLocator(ttft_dist_interval))
    axes[0, 1].set_xlim(left=0) # Force X-axis to start from 0
    axes[0, 1].grid(True)

    # 3. Bottleneck Analysis (TTFT vs ITL)
    sns.scatterplot(data=df, x='ttft_ms', y='avg_itl_ms', hue='model', palette=palette, hue_order=unique_models, ax=axes[1, 0])
    axes[1, 0].set_title('3. Bottleneck Analysis: Scheduling(TTFT) vs Compute(ITL)')
    axes[1, 0].set_ylim(bottom=0) # Ensure 0 is visible

    # Auto-adjust X-axis (TTFT) interval
    axes[1, 0].xaxis.set_major_locator(MultipleLocator(ttft_interval))

    # Auto-adjust Y-axis (ITL) interval
    itl_range = df['avg_itl_ms'].max() - df['avg_itl_ms'].min()
    itl_interval = get_smart_interval(itl_range, target_ticks=10)
    axes[1, 0].yaxis.set_major_locator(MultipleLocator(itl_interval))
    axes[1, 0].grid(True)

    # 4. Total Latency Comparison (Total Latency by Model)
    sns.boxplot(data=df, x='model', y='total_latency_ms', hue='model', palette=palette, order=unique_models, ax=axes[1, 1], legend=False)
    axes[1, 1].set_title('4. Total Latency Overview per Model')

    # Auto-adjust Y-axis (Total Latency) interval
    total_latency_range = df['total_latency_ms'].max() - df['total_latency_ms'].min()
    total_latency_interval = get_smart_interval(total_latency_range, target_ticks=10)
    axes[1, 1].yaxis.set_major_locator(MultipleLocator(total_latency_interval))
    axes[1, 1].grid(True)

    plt.tight_layout()
    # plt.show()
    print(f"Saving figure: {output_name}.png")
    plt.savefig(f'{output_name}.png')
    plt.close()