import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Style setup
def _use_first_available_style(styles):
    available = set(plt.style.available)
    for s in styles:
        if s in available:
            plt.style.use(s)
            return s
    return None

_use_first_available_style([
    "seaborn-v0_8-white", "seaborn-white", "seaborn-v0_8-paper", "seaborn-paper", "default",
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
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "axes.axisbelow": True,
})

# Data
model_sizes = [0.4, 1, 7, 26]
x_positions = np.arange(len(model_sizes))

baselines = {
    'floq': {
        'mean': [33, 31, 27, 27], 'std': [5, 5, 5, 4],
        'color': '#7F7F7F', 'marker': 's',
    },
    'FQL': {
        'mean': [27, 26, 25, 24], 'std': [5, 4, 5, 5],
        'color': '#9E9E9E', 'marker': '^',
    },
    'PAC': {
        'mean': [26, 23, 21, 23], 'std': [9, 8, 8, 5],
        'color': '#B0B0B0', 'marker': 'D',
    },
    'Transformer w/o TQL': {
        'mean': [27, 26, 30, 25], 'std': [8, 7, 8, 4],
        'color': '#6B6B6B', 'marker': 'P',
    },
}

tql = {
    'mean': [28, 33, 36, 40], 'std': [11, 8, 7, 7],
    'color': '#ff7f0e', 'marker': 'o',
}

# Pre-compute relative improvements
def compute_improvement(mean_vals):
    mean = np.array(mean_vals, dtype=float)
    first_val = mean[0]
    if first_val != 0:
        return ((mean / first_val) - 1) * 100
    return np.zeros_like(mean)

baseline_improvements = {}
for name, data in baselines.items():
    baseline_improvements[name] = compute_improvement(data['mean'])

tql_improvement = compute_improvement(tql['mean'])

# Compute average baseline degradation at 26M
baseline_final_vals = [baseline_improvements[name][-1] for name in baselines]
avg_baseline_drop = np.mean(baseline_final_vals)

# Set up figure — taller to avoid clipping
fig, ax = plt.subplots(1, 1, figsize=(6, 3.8), constrained_layout=False)
fig.suptitle("TQL unlocks scaling of value functions in RL", fontsize=13, fontweight='semibold', y=0.97)

# Configure axes
ax.set_xlabel('Critic Model Size', fontsize=12, color='black')
ax.set_ylabel('Relative Improvement (%)', fontsize=12, color='black')
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{s:.1f}M' if s < 1 else f'{int(s)}M' for s in model_sizes])
ax.set_xlim(-0.3, len(model_sizes) - 0.7)
ax.set_ylim(-30, 60)
ax.set_yticks(np.arange(-20, 61, 20))
ax.axhline(0.0, color='#B0B0B0', linestyle='--', linewidth=1.0, zorder=1, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['left'].set_color('#222222')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['bottom'].set_color('#222222')
ax.tick_params(axis='both', which='major', length=4, width=1.0, color='#222222')
ax.tick_params(axis='both', which='minor', length=0)
ax.grid(False)

n_points = len(model_sizes)

# Animation timing (slower)
points_between = 21   # frames between points (1.2x faster drawing)
hold_start = 15       # hold at start
hold_end = 80         # long hold at end to read summary text

total_draw_frames = points_between * (n_points - 1) + 1
# Extra frames for the summary text fade-in
summary_fade_frames = 15
total_frames = hold_start + total_draw_frames + summary_fade_frames + hold_end

def interp_data(improvements, frac_points):
    """Interpolate data up to frac_points (float, 0 to n_points-1)."""
    n = int(np.floor(frac_points)) + 1
    n = min(n, len(improvements))
    frac = frac_points - int(np.floor(frac_points))

    xs = x_positions[:n].tolist()
    ys = improvements[:n].tolist()

    if n < len(improvements) and frac > 0:
        next_x = x_positions[n]
        next_y = improvements[n]
        interp_x = x_positions[n-1] + frac * (next_x - x_positions[n-1])
        interp_y = ys[-1] + frac * (next_y - ys[-1])
        xs.append(interp_x)
        ys.append(interp_y)

    return np.array(xs), np.array(ys)

# Create line objects
baseline_lines = {}
baseline_markers = {}
for name, data in baselines.items():
    line, = ax.plot([], [], linewidth=1.4, color=data['color'], alpha=0.98, zorder=2)
    marker, = ax.plot([], [], marker=data['marker'], linestyle='None', color=data['color'],
                      markersize=6, markerfacecolor=data['color'], markeredgewidth=0.9,
                      markeredgecolor='white', zorder=2, label=name)
    baseline_lines[name] = line
    baseline_markers[name] = marker

tql_line, = ax.plot([], [], linewidth=1.4, color=tql['color'], alpha=0.98, zorder=3)
tql_marker, = ax.plot([], [], marker=tql['marker'], linestyle='None', color=tql['color'],
                      markersize=6, markerfacecolor=tql['color'], markeredgewidth=0.9,
                      markeredgecolor='white', zorder=3, label='TQL')

# Dynamic text annotations
dynamic_texts = []

# Legend
handles = list(baseline_markers.values()) + [tql_marker]
labels = list(baselines.keys()) + ['TQL']
fig.legend(handles, labels, loc='lower center', ncol=len(handles), frameon=False,
           fontsize=9, handlelength=2.0, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(bottom=0.22, top=0.88)

def init():
    for name in baselines:
        baseline_lines[name].set_data([], [])
        baseline_markers[name].set_data([], [])
    tql_line.set_data([], [])
    tql_marker.set_data([], [])
    return []

def animate(frame):
    # Remove old dynamic texts
    for t in dynamic_texts:
        t.remove()
    dynamic_texts.clear()

    # Phase 1: hold at start
    # Phase 2: draw lines
    # Phase 3: show summary text
    # Phase 4: hold at end

    draw_start = hold_start
    draw_end = hold_start + total_draw_frames
    summary_start = draw_end
    summary_end = summary_start + summary_fade_frames

    # Calculate drawing progress
    if frame < draw_start:
        frac_points = 0.0
        show_summary = False
        summary_alpha = 0.0
    elif frame < draw_end:
        draw_frame = frame - draw_start
        frac_points = draw_frame / points_between
        frac_points = min(frac_points, float(n_points - 1))
        show_summary = False
        summary_alpha = 0.0
    else:
        frac_points = float(n_points - 1)
        show_summary = True
        if frame < summary_end:
            summary_alpha = (frame - summary_start) / summary_fade_frames
        else:
            summary_alpha = 1.0

    # Update baselines
    for name, data in baselines.items():
        imp = baseline_improvements[name]
        xs, ys = interp_data(imp, frac_points)
        baseline_lines[name].set_data(xs, ys)
        n_full = int(np.floor(frac_points)) + 1
        n_full = min(n_full, len(imp))
        baseline_markers[name].set_data(x_positions[:n_full], imp[:n_full])

    # Update TQL
    xs, ys = interp_data(tql_improvement, frac_points)
    tql_line.set_data(xs, ys)
    n_full = int(np.floor(frac_points)) + 1
    n_full = min(n_full, len(tql_improvement))
    tql_marker.set_data(x_positions[:n_full], tql_improvement[:n_full])

    # TQL percentage annotations for revealed points
    for i in range(1, n_full):
        val = tql_improvement[i]
        sign = '+' if val >= 0 else ''
        t = ax.text(x_positions[i], val + 2.5, f'{sign}{val:.0f}%',
                    fontsize=9, color=tql['color'], va='bottom', ha='center',
                    fontweight='normal')
        dynamic_texts.append(t)

    # Summary annotation after drawing is complete
    if show_summary:
        # Invisible 2-line text to create correctly sized box
        t_box = ax.text(
            0.05, 0.93,
            "Prior methods: ~10% avg. decrease\nTQL: +43% improvement",
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            ha='left', va='top',
            color=(0, 0, 0, 0),  # fully transparent text
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor='white',
                edgecolor='#E67E22',
                linewidth=1.5,
                alpha=summary_alpha * 0.95,
            ),
            zorder=10,
        )
        dynamic_texts.append(t_box)

        # Grey first line (same position as invisible text)
        t_grey = ax.text(
            0.05, 0.93,
            "Prior methods: ~10% avg. decrease",
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            ha='left', va='top',
            alpha=summary_alpha,
            color='#7F7F7F',
            zorder=11,
        )
        dynamic_texts.append(t_grey)

        # Orange second line (offset down by one line height)
        t_orange = ax.text(
            0.05, 0.85,
            "TQL: +43% improvement",
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            ha='left', va='top',
            alpha=summary_alpha,
            color='#E67E22',
            zorder=11,
        )
        dynamic_texts.append(t_orange)

    return []

print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=total_frames, interval=70, blit=False)

out_path = '/Users/johnsonhung/Desktop/Research/iliad/tql /website/static/figures/scale_animated.gif'
anim.save(out_path, writer='pillow', fps=14,
          savefig_kwargs={'facecolor': 'white'})
print(f"Saved animated GIF to {out_path}")
print(f"Total frames: {total_frames}, FPS: 14, Duration: {total_frames/14:.1f}s")
