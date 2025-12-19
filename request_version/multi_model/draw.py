import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

TARGET_TTFT=5000 # ms

# Read data
cases = ["round_robin", "zipfian", "bursty"]
dfs = {}
all_models = set()

# Step 1: Read all data to determine all appearing models
for case in cases:
    file_name = f"benchmark_results_{case}"
    try:
        df = pd.read_csv(f'{file_name}.csv')
        # Preprocessing: Normalize start time to start from 0
        df['relative_start_time'] = df['start_time'] - df['start_time'].min()
        dfs[case] = df
        all_models.update(df['model'].unique())
    except FileNotFoundError:
        print(f"Warning: {file_name}.csv not found")

# Set unified colors and order
unique_models = sorted(list(all_models))
palette = dict(zip(unique_models, sns.color_palette("tab10", len(unique_models))))

for case in cases:
    if case not in dfs:
        continue
        
    df = dfs[case]
    file_name = f"benchmark_results_{case}"

    # Set up canvas
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. System Stability (Time vs TTFT)
    sns.scatterplot(data=df, x='relative_start_time', y='ttft_ms', hue='model', palette=palette, hue_order=unique_models, ax=axes[0, 0])
    axes[0, 0].set_title('1. System Stability: TTFT over Time')
    axes[0, 0].set_xlabel('Time Elapsed (s)')
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(10))  # Time 10s per unit
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(1000)) # TTFT 1000ms per unit
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
    axes[0, 1].xaxis.set_major_locator(MultipleLocator(1000)) # TTFT 1000ms per unit
    axes[0, 1].set_xlim(left=0) # Force X-axis to start from 0
    axes[0, 1].grid(True)

    # 3. Bottleneck Analysis (TTFT vs ITL)
    sns.scatterplot(data=df, x='ttft_ms', y='avg_itl_ms', hue='model', palette=palette, hue_order=unique_models, ax=axes[1, 0])
    axes[1, 0].set_title('3. Bottleneck Analysis: Scheduling(TTFT) vs Compute(ITL)')
    axes[1, 0].set_ylim(bottom=0) # Ensure 0 is visible
    axes[1, 0].xaxis.set_major_locator(MultipleLocator(1000)) # TTFT 1000ms per unit
    axes[1, 0].grid(True)

    # 4. Total Latency Comparison (Total Latency by Model)
    sns.boxplot(data=df, x='model', y='total_latency_ms', hue='model', palette=palette, order=unique_models, ax=axes[1, 1], legend=False)
    axes[1, 1].set_title('4. Total Latency Overview per Model')
    axes[1, 1].yaxis.set_major_locator(MultipleLocator(1000)) # TTFT 1000ms per unit
    axes[1, 1].grid(True)

    plt.tight_layout()
    # plt.show()
    print(f"save figure {file_name}.png")
    plt.savefig(f'{file_name}.png')
    plt.close()