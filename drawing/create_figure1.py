import matplotlib
# Force a non-interactive backend for reliable rendering in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality figures (clean, grid-free)
# Matplotlib style names differ across versions (e.g., `seaborn-white` vs `seaborn-v0_8-white`),
# so pick the first available style to avoid runtime errors.
def _use_first_available_style(styles: list[str]) -> str | None:
    available = set(plt.style.available)
    for s in styles:
        if s in available:
            plt.style.use(s)
            return s
    return None

_style_used = _use_first_available_style([
    "seaborn-v0_8-white",
    "seaborn-white",
    "seaborn-v0_8-paper",
    "seaborn-paper",
    "default",
])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,  # remove grey grid/background lines
    "axes.axisbelow": True,
})

# Data configuration
model_sizes = [0.4, 1, 7, 26]  # Model sizes in millions

# Baselines (all grey shades)
baselines = {
    'floq': {
        'mean': [33, 31, 27, 27],
        'std': [5, 5, 5, 4],
        'color': '#7F7F7F',  # grey
        'marker': 's',
    },
    'FQL': {
        'mean': [27, 26, 25, 24],
        'std': [5, 4, 5, 5],
        'color': '#9E9E9E',  # light grey
        'marker': '^',
    },
    'PAC': {
        'mean': [26, 23, 21, 23],  # Empty first value
        'std': [9, 8, 8, 5],
        'color': '#B0B0B0',  # lighter grey
        'marker': 'D',
    },
    'Transformer w/o TQL': {
        'mean': [27, 26, 30, 25],
        'std': [8, 7, 8, 4],
        'color': '#6B6B6B',  # dark grey
        'marker': 'P',
    },
}

# TQL method
tql = {
    'mean': [28, 33, 36, 40],
    'std': [11, 8, 7, 7],
    'color': '#ff7f0e',  # orange
    'marker': 'o',
}

# Configuration
FIGURE_TITLE = "TQL unlocks scaling of value functions in RL"

# Create a single figure (one panel)
fig, ax1 = plt.subplots(
    1, 1,
    figsize=(6, 3),
    constrained_layout=False,
)
fig.suptitle(FIGURE_TITLE, fontsize=13, fontweight='semibold', y=1.06)

# Function to plot data on an axis
def plot_scaling(ax, model_sizes, baselines, tql):
    model_sizes = np.array(model_sizes)
    x_positions = np.arange(len(model_sizes))
    
    # Normalize all data by subtracting first point (delta improvement)
    all_improvements = []
    
    # Plot all baselines (grey shades)
    for name, baseline_data in baselines.items():
        mean = np.array(baseline_data['mean'])
        color = baseline_data['color']
        marker = baseline_data['marker']
        
        # Mask NaN values
        valid = ~np.isnan(mean)
        if not np.any(valid):
            continue
        
        # Calculate relative improvement as ratio: (value / first_value) - 1, then * 100 for percentage
        first_valid_idx = np.where(valid)[0][0]
        first_val = mean[first_valid_idx]
        if first_val != 0:
            mean_improvement = ((mean[valid] / first_val) - 1) * 100
        else:
            mean_improvement = np.zeros_like(mean[valid])
        all_improvements.extend(mean_improvement)

        # Plot line with markers (no std bands)
        ax.plot(
            x_positions[valid], mean_improvement,
            marker=marker,
            linewidth=1.4,
            label=name,
            color=color,
            markersize=6,
            markerfacecolor=color,
            markeredgewidth=0.9,
            markeredgecolor='white',
            alpha=0.98,
            zorder=2,
        )
    
    # Plot TQL (orange) - calculate relative improvement as ratio: (value / first_value) - 1, then * 100 for percentage
    mean_tql = np.array(tql['mean'])
    first_val_tql = mean_tql[0]
    if first_val_tql != 0:
        mean_tql_improvement = ((mean_tql / first_val_tql) - 1) * 100
    else:
        mean_tql_improvement = np.zeros_like(mean_tql)
    all_improvements.extend(mean_tql_improvement)
    
    # Plot line with markers (no std bands)
    ax.plot(
        x_positions, mean_tql_improvement,
        marker=tql['marker'],
        linewidth=1.4,
        label='TQL',
        color=tql['color'],
        markersize=6,
        markerfacecolor=tql['color'],
        markeredgewidth=0.9,
        markeredgecolor='white',
        alpha=0.98,
        zorder=3,
    )

    # Add improvement annotations for TQL (percentage improvement values)
    y_range = max(all_improvements) - min(all_improvements) if all_improvements else 30
    offset = y_range * 0.05

    for i, (x_pos, tql_improvement) in enumerate(zip(x_positions, mean_tql_improvement)):
        if i > 0:  # Skip first point (always 0)
            sign = '+' if tql_improvement >= 0 else ''
            ax.text(x_pos, tql_improvement + offset, f'{sign}{tql_improvement:.0f}%',
                   fontsize=9, color=tql['color'], va='bottom', ha='center',
                   fontweight='normal')

    # Set labels
    ax.set_xlabel('Critic Model Size', fontsize=12, color='black')
    ax.set_ylabel('Relative Improvement (%)', fontsize=12, color='black')

    # Set x-axis ticks with equal spacing
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{size:.1f}M' if size < 1 else f'{int(size)}M'
                        for size in model_sizes])

    # Set y-axis range - auto-calculate based on improvements
    if all_improvements:
        y_min = min(all_improvements)
        y_max = max(all_improvements)
        y_range = y_max - y_min
        y_min = y_min - 0.1 * y_range
        y_max = y_max + 0.1 * y_range
        # Round to nice numbers
        y_min = np.floor(y_min / 5) * 5
        y_max = 60 # np.ceil(y_max / 5) * 5
    else:
        y_min, y_max = -10, 20
    
        ax.set_ylim(y_min, y_max)
    
    # Set y ticks - more sparse
    if (y_max - y_min) <= 30:
        tick_step = 10
    elif (y_max - y_min) <= 60:
        tick_step = 20
    else:
        tick_step = 25
    ax.set_yticks(np.arange(y_min, y_max + 1, tick_step))
    
    # Add reference line at 0
    ax.axhline(0.0, color='#B0B0B0', linestyle='--', linewidth=1.0, zorder=1, alpha=0.5)

    # Ensure no grid/background lines
    ax.grid(False)

    # Set x limits for equal spacing
    ax.set_xlim(-0.3, len(model_sizes) - 0.7)

    # Improve spine appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#222222')
    ax.tick_params(axis='both', which='major', length=4, width=1.0, color='#222222')
    ax.tick_params(axis='both', which='minor', length=0)

# Plot the figure
plot_scaling(ax1, model_sizes, baselines, tql)

# Create legend at the bottom in horizontal layout
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=len(handles),
    frameon=False,
    fontsize=9,
    handlelength=2.0,
    bbox_to_anchor=(0.5, -0.05),
)

# Add extra bottom margin for the legend
fig.subplots_adjust(bottom=0.25)

# Save the figure with high quality
plt.savefig('figure1.png', dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
plt.savefig('figure1.pdf', bbox_inches='tight', pad_inches=0.05, facecolor='white')

print("✓ Figure saved as 'figure1.png' and 'figure1.pdf'")
print("✓ Figure size: 6×3 inches")
print("✓ Resolution: 300 DPI")
print("✓ Enhanced features:")
print("  • Multiple baselines with different grey shades")
print("  • Uncertainty bands (shaded regions for ±1 std)")
print("  • Annotated improvements for TQL method")
print("  • Legend (upper-left)")
print("  • Clean background (no grid lines)")
print("  • Constrained layout for optimal spacing")
print("  • Cleaner spines (removed top/right)")
