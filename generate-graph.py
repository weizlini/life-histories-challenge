import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import json
import random
import string
import sys

# Silence overflow warnings from steep exponential terms
np.seterr(over='ignore')

# Generate high-resolution time data
t = np.linspace(0, 110, 3000)

# Seed for reproducibility so the graph and JSON match every time
np.random.seed(42)
random.seed(42)

lines = []
labels = []
ground_truth = {
    "species_count": 35,
    "k_selected_count_total": 10,
    "r_selected_count_total": 25,
    "k_selected_count_valid": 9,
    "r_selected_count_valid": 24,
    "decoys_to_ignore": [],
    "classifications": {}
}

# Generate 35 unique random alphanumeric codes (e.g., "XK-42")
unique_codes = set()
while len(unique_codes) < 35:
    code = "".join(random.choices(string.ascii_uppercase, k=2)) + "-" + "".join(random.choices(string.digits, k=2))
    unique_codes.add(code)
unique_codes = sorted(list(unique_codes))
random.shuffle(unique_codes)

# 1. TIGHTER SPACING (Middle ground: ~49 units apart at t=0, distinct for humans but dense in the middle)
all_target_starts = list(np.linspace(50, 1750, 35))
random.shuffle(all_target_starts)
k_starts = all_target_starts[:10]
r_starts = all_target_starts[10:]

# Pick exactly one K-selected and one r-selected to be the >20% gap decoys
decoy_k_idx = random.randint(0, 9)
decoy_r_idx = random.randint(0, 24)


def inject_missing_data(pop_array, time_array, is_decoy=False):
    """Simulates missed ecological surveys by injecting np.nan gaps."""
    if is_decoy:
        # Force exactly 27.5% missing data via 5 perfectly spaced 5.5-year gaps
        gap_starts = [10, 28, 46, 64, 82]
        for st in gap_starts:
            gap_mask = (time_array >= st) & (time_array <= st + 5.5)
            pop_array[gap_mask] = np.nan
    else:
        # Normal lines get 2-4 small gaps (max ~7% missing data)
        num_gaps = np.random.randint(2, 5)
        for _ in range(num_gaps):
            gap_start = np.random.uniform(5, 95)
            gap_len = np.random.uniform(0.8, 2.2)
            gap_mask = (time_array >= gap_start) & (time_array <= gap_start + gap_len)
            pop_array[gap_mask] = np.nan
    return pop_array


def add_shatter_jitter(pop_array, time_array):
    """Applies continuous high-frequency seasonal wobble (strictly 0 at t=0)."""
    ramp = np.minimum(1.0, time_array / 4.0)
    envelope = (0.5 + 0.3 * np.sin(time_array * np.random.uniform(0.02, 0.08)) +
                0.2 * np.cos(time_array * np.random.uniform(0.01, 0.05))) * ramp
    wobble1 = np.random.uniform(12, 24) * np.sin(time_array * np.random.uniform(3.0, 5.5))
    wobble2 = np.random.randint(2, 8) * np.sin(time_array * np.random.uniform(0.8, 2.0))
    pop_array += (wobble1 + wobble2) * envelope
    return pop_array


def add_sectional_wobble(pop_array, time_array):
    """Adds localized bursts of wobble to otherwise smooth curves so they aren't uniformly straight."""
    ramp = np.minimum(1.0, time_array / 4.0)
    num_patches = np.random.randint(1, 3)
    for _ in range(num_patches):
        center = np.random.uniform(18, 88)
        span = np.random.uniform(4.5, 9.0)
        # Gaussian window restricts the wobble to a specific time section
        window = np.exp(-((time_array - center) ** 2) / (2 * span ** 2)) * ramp
        wobble = np.random.uniform(10, 22) * np.sin(time_array * np.random.uniform(2.5, 5.0))
        pop_array += wobble * window
    return pop_array


# ==========================================
# POPULATION MATH (35 Species)
# ==========================================

# --- 10 K-selected species: Tighter vertical domain (180 to 1750) ---
k_ends = list(np.linspace(180, 1750, 10))
random.shuffle(k_ends)

for i in range(10):
    name = f"Species {unique_codes[i]}"
    labels.append(name)
    ground_truth["classifications"][name] = "K-selected"

    is_decoy = (i == decoy_k_idx)
    if is_decoy:
        ground_truth["decoys_to_ignore"].append(name)

    start_pop = k_starts[i]
    end_pop = k_ends[i]
    midpoint = np.random.uniform(22, 65)
    rate = np.random.uniform(0.06, 0.14)

    # Normalized logistic curve so pop[0] == start_pop exactly
    raw_logistic = 1.0 / (1.0 + np.exp(-rate * (t - midpoint)))
    raw_0 = 1.0 / (1.0 + np.exp(-rate * (0.0 - midpoint)))
    norm_logistic = (raw_logistic - raw_0) / (1.0 - raw_0)
    pop = start_pop + (end_pop - start_pop) * norm_logistic

    # Gentle carrying-capacity adjustment
    if random.random() > 0.5:
        shift_time = np.random.uniform(50, 80)
        shift_amp = np.random.uniform(-180, 180)
        shift_rate = np.random.uniform(0.06, 0.12)
        s_raw = 1.0 / (1.0 + np.exp(-shift_rate * (t - shift_time)))
        s_0 = 1.0 / (1.0 + np.exp(-shift_rate * (0.0 - shift_time)))
        pop += shift_amp * (s_raw - s_0)

    # 50% get full shatter jitter; the "smooth" 50% get localized sectional wobble
    if random.random() > 0.5:
        pop = add_shatter_jitter(pop, t)
    else:
        pop = add_sectional_wobble(pop, t)

    pop = np.clip(pop, 10, None)
    pop[0] = start_pop  # Lock exact t=0 value
    pop_full = pop.copy()
    pop_gaps = inject_missing_data(pop.copy(), t, is_decoy)
    lines.append((pop_gaps, pop_full))

# --- 25 r-selected species: Compressed baselines (60 to 1050) for denser crossings ---
r_baselines = list(np.linspace(60, 1050, 25))
random.shuffle(r_baselines)

for i in range(25):
    name = f"Species {unique_codes[i + 10]}"
    labels.append(name)
    ground_truth["classifications"][name] = "r-selected"

    is_decoy = (i == decoy_r_idx)
    if is_decoy:
        ground_truth["decoys_to_ignore"].append(name)

    target_start = float(r_starts[i])
    base_pop = float(r_baselines[i])

    # Smooth transition from exact target_start at t=0 to the species' baseline
    trans_mid = np.random.uniform(8, 22)
    trans_rate = np.random.uniform(0.18, 0.35)
    raw_trans = 1.0 / (1.0 + np.exp(-trans_rate * (t - trans_mid)))
    raw_trans_0 = 1.0 / (1.0 + np.exp(-trans_rate * (0.0 - trans_mid)))
    norm_trans = (raw_trans - raw_trans_0) / (1.0 - raw_trans_0)
    pop = target_start + (base_pop - target_start) * norm_trans

    # 50% get full shatter jitter; the other 50% get localized sectional wobble
    if random.random() > 0.5:
        pop = add_shatter_jitter(pop, t)
    else:
        pop = add_sectional_wobble(pop, t)

    # 2 to 4 unmistakable boom-and-bust spikes (>550 prominence)
    num_spikes = np.random.randint(2, 5)
    windows = np.linspace(14, 92, num_spikes + 1)
    for w_idx in range(num_spikes):
        spike_time = np.random.uniform(windows[w_idx] + 2, windows[w_idx + 1] - 2)
        spike_height = np.random.uniform(600, 1100)
        width = np.random.uniform(2.5, 7.5)

        spike_curve = spike_height * np.exp(-((t - spike_time) ** 2) / width)
        spike_0 = spike_height * np.exp(-((0.0 - spike_time) ** 2) / width)
        pop += (spike_curve - spike_0)

    pop = np.clip(pop, 10, None)
    pop[0] = target_start  # Lock exact t=0 value
    pop_full = pop.copy()
    pop_gaps = inject_missing_data(pop.copy(), t, is_decoy)
    lines.append((pop_gaps, pop_full))

combined = list(zip(labels, lines))
random.shuffle(combined)
labels, lines = zip(*combined)

max_pop_value = max([np.nanmax(pop_full) for pop_gaps, pop_full in lines])
y_limit = max(1950.0, max_pop_value * 1.05)

with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)
print("Saved ground truth to ground_truth.json")

# ==========================================
# PLOTTING THE TWO-PANEL GRAPH
# ==========================================

fig = plt.figure(figsize=(15, 18))
gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.5], hspace=0.32)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], polar=True)

start_vals = [pop_full[0] for pop_gaps, pop_full in lines]
sorted_indices = np.argsort(start_vals)

# --- Top Panel: Trajectories with Dotted Sampling Gaps ---
for pop_gaps, pop_full in lines:
    plot_mask = t <= 100
    z = random.randint(10, 100)

    # Isolate ONLY the missing gap segments (plus 1 boundary point on each side to connect cleanly)
    nan_mask = np.isnan(pop_gaps)
    gap_draw_mask = np.convolve(nan_mask.astype(int), [1, 1, 1], mode='same') > 0
    pop_missing = np.where(gap_draw_mask, pop_full, np.nan)

    # Plot dotted line strictly inside the sampling gaps
    ax1.plot(t[plot_mask], pop_missing[plot_mask], color='black', linewidth=1.1,
             linestyle=':', zorder=z)
    # Plot solid line where survey data exists
    ax1.plot(t[plot_mask], pop_gaps[plot_mask], color='black', linewidth=1.0,
             solid_joinstyle='round', zorder=z + 1)

# Evenly spaced label positions on the left margin
text_y_positions = np.linspace(25, y_limit * 0.96, len(lines))

for rank, idx in enumerate(sorted_indices):
    actual_y = start_vals[idx]
    text_y = text_y_positions[rank]
    label = labels[idx]
    text_x = -14
    ax1.text(text_x, text_y, label, va='center', ha='right', fontsize=8, fontweight='bold', zorder=150)
    # Leader line connects to a guaranteed unique, non-overlapping origin point at t=0
    ax1.plot([text_x + 0.8, 0], [text_y, actual_y], color='gray', linewidth=0.7, zorder=140)
    # Small origin dot at t=0 so human reviewers can see each distinct starting point clearly
    ax1.scatter([0], [actual_y], color='black', s=10, zorder=160)

# Legend for the dotted sampling gaps
gap_legend_line = mlines.Line2D([], [], color='black', linewidth=1.2, linestyle=':', label='Sampling Gap (Inferred)')
ax1.legend(handles=[gap_legend_line], loc='upper left', bbox_to_anchor=(0.18, 0.99), fontsize=10, frameon=False)

# Note defines the life-history rule qualitatively without giving away the numeric value
ax1.text(0.99, 0.97,
         "* Note: Species with >20% survey gaps are not valid.\n"
         "  r-selected species exhibit boom-and-bust spikes; K-selected follow logistic curves.",
         transform=ax1.transAxes, ha='right', va='top', fontsize=10, color='black', fontstyle='italic',
         bbox=dict(facecolor='white', edgecolor='black', pad=5), zorder=250)

ax1.set_xlim(-32, 100)
ax1.set_ylim(0, y_limit)
ax1.set_ylabel("Population Size", fontsize=14, fontweight='bold', labelpad=10)
ax1.set_title("Population Dynamics of 35 Canadian Arctic Nature Reserve Introduced Species", fontsize=16,
              fontweight='bold', pad=20)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)

# --- Bottom Panel: Polar Data Availability Chart (With Radial Non-Overlapping Labels) ---
completeness_scores = []
for pop_gaps, _ in lines:
    plot_mask = t <= 100
    total_points = np.sum(plot_mask)
    valid_points = np.sum(~np.isnan(pop_gaps[plot_mask]))
    completeness_scores.append((valid_points / total_points) * 100)

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles_closed = np.concatenate((angles, [angles[0]]))
scores_closed = np.concatenate((completeness_scores, [completeness_scores[0]]))

# Plot the data (monochromatic)
ax2.plot(angles_closed, scores_closed, color='black', linewidth=1.0, zorder=3)
ax2.scatter(angles, completeness_scores, color='black', s=20, zorder=4)

ax2.set_ylim(0, 105)
ax2.set_yticks([25, 50, 75, 100])
ax2.set_yticklabels(['25', '50', '75', '100'], color='gray', fontsize=8)
ax2.set_xticks(angles)
ax2.set_xticklabels([])  # Hide default horizontal labels to prevent pole collisions

# Place rotated radial labels along each spoke so zero labels ever touch or overlap
for label_text, angle in zip(labels, angles):
    angle_deg = np.degrees(angle)
    # Flip text on the left half of the circle so it reads naturally without being upside-down
    if 90 < angle_deg < 270:
        rotation = angle_deg + 180
        ha = 'right'
    else:
        rotation = angle_deg
        ha = 'left'

    ax2.text(angle, 113, label_text, rotation=rotation, ha=ha, va='center',
             rotation_mode='anchor', fontsize=7.5, fontweight='bold', clip_on=False)

# Increased pad from 45 to 85 so the title clears the top vertical spoke labels cleanly
ax2.set_title("Survey of Data Captured", fontsize=12, fontweight='bold', pad=85)

plt.tight_layout()
plt.savefig("population_dynamics.png", dpi=300, bbox_inches='tight')
print("Saved clean two-panel visual to population_dynamics.png")

# ==========================================
# PLOTTING THE ANNOTATED GRAPH
# ==========================================
for i, (pop_gaps, pop_full) in enumerate(lines):
    label = labels[i]
    if ground_truth["classifications"][label] == "K-selected":
        plot_mask = t <= 100
        visible_indices = np.where(plot_mask)[0]
        if len(visible_indices) > 0:
            dot_idx = random.choice(visible_indices[500:-500])
            ax1.plot(t[dot_idx], pop_full[dot_idx], 'ro', markersize=6, zorder=300)

plt.savefig("population_dynamics_annotated.png", dpi=300, bbox_inches='tight')
print("Saved annotated two-panel visual to population_dynamics_annotated.png")

print("Visual generation complete. Exiting.")
sys.exit(0)