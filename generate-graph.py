import numpy as np
import matplotlib.pyplot as plt
import json
import random
import string
import sys

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
unique_codes = list(unique_codes)
random.shuffle(unique_codes)

# Expanded vertical space to create sharper, more distinct 'X' intersections
all_target_starts = random.sample(range(20, 1400), 35)
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
            gap_len = np.random.uniform(0.5, 2.0)
            gap_mask = (time_array >= gap_start) & (time_array <= gap_start + gap_len)
            pop_array[gap_mask] = np.nan
    return pop_array


def add_shatter_jitter(pop_array, time_array):
    """Applies high-frequency seasonal wobble with a dynamic amplitude envelope."""
    envelope = 0.5 + 0.3 * np.sin(time_array * np.random.uniform(0.02, 0.08)) + 0.2 * np.cos(
        time_array * np.random.uniform(0.01, 0.05))
    wobble1 = np.random.uniform(15, 35) * np.sin(time_array * np.random.uniform(3.0, 6.0))
    wobble2 = np.random.randint(2, 10) * np.cos(time_array * np.random.uniform(0.8, 2.0))
    pop_array += (wobble1 + wobble2) * envelope
    return pop_array


# ==========================================
# POPULATION MATH (35 Species)
# ==========================================

for i in range(10):
    name = f"Species {unique_codes[i]}"
    labels.append(name)
    ground_truth["classifications"][name] = "K-selected"

    is_decoy = (i == decoy_k_idx)
    if is_decoy:
        ground_truth["decoys_to_ignore"].append(name)

    start_pop = k_starts[i]
    end_pop = np.random.randint(100, 1400)  # Scaled up to match new space
    midpoint = np.random.randint(20, 80)
    rate = np.random.uniform(0.05, 0.2)

    pop = start_pop + (end_pop - start_pop) / (1 + np.exp(-rate * (t - midpoint)))

    # VERTICAL crashes
    if random.random() > 0.5:
        crash_time = np.random.uniform(20, 80)
        crash_idx = int((crash_time / 110.0) * 3000)
        if crash_idx < 3000:
            current_level = pop[crash_idx]
            crash_depth = np.random.uniform(0.4, 0.7) * current_level
            crash_steepness = np.random.uniform(5.0, 15.0)
            pop -= crash_depth / (1 + np.exp(-crash_steepness * (t - crash_time)))
            recovery_rate = np.random.uniform(0.02, 0.08)
            pop += (crash_depth * np.random.uniform(0.3, 0.8)) / (1 + np.exp(-recovery_rate * (t - (crash_time + 15))))

    if random.random() > 0.5:
        pop = add_shatter_jitter(pop, t)

    pop = np.clip(pop, 5, None)
    pop_full = pop.copy()
    pop_gaps = inject_missing_data(pop.copy(), t, is_decoy)
    lines.append((pop_gaps, pop_full))

for i in range(25):
    name = f"Species {unique_codes[i + 10]}"
    labels.append(name)
    ground_truth["classifications"][name] = "r-selected"

    is_decoy = (i == decoy_r_idx)
    if is_decoy:
        ground_truth["decoys_to_ignore"].append(name)

    target_start = float(r_starts[i])

    if target_start > 250:
        base_pop = np.random.uniform(10.0, 180.0)
        crash_midpoint = np.random.uniform(5, 25)
        crash_rate = np.random.uniform(0.2, 0.5)
        height = (target_start - base_pop) * (1 + np.exp(-crash_rate * crash_midpoint))
        pop = base_pop + height / (1 + np.exp(crash_rate * (t - crash_midpoint)))
        spike_start = 30
    else:
        if random.random() > 0.5:
            fake_k_end = np.random.uniform(target_start + 50, target_start + 400)
            fake_mid = np.random.uniform(20, 50)
            fake_rate = np.random.uniform(0.05, 0.15)
            pop = target_start + (fake_k_end - target_start) / (1 + np.exp(-fake_rate * (t - fake_mid)))
            spike_start = np.random.uniform(55, 75)
        else:
            base_pop = target_start
            pop = np.full_like(t, base_pop, dtype=float)
            early_spike = np.random.uniform(5, 15)
            early_height = np.random.randint(600, 1400)  # Scaled up
            early_width = np.random.uniform(2, 6)
            pop += early_height * np.exp(-((t - early_spike) ** 2) / early_width)
            spike_start = 30

    if random.random() > 0.5:
        pop = add_shatter_jitter(pop, t)

    num_spikes = np.random.randint(2, 6)
    for _ in range(num_spikes):
        spike_time = np.random.uniform(spike_start, 110)
        spike_height = np.random.randint(600, 1400)
        width = np.random.uniform(30, 80) if random.random() > 0.7 else np.random.uniform(2, 8)
        pop += spike_height * np.exp(-((t - spike_time) ** 2) / width)

    pop = np.clip(pop, 5, None)
    pop_full = pop.copy()
    pop_gaps = inject_missing_data(pop.copy(), t, is_decoy)
    lines.append((pop_gaps, pop_full))

combined = list(zip(labels, lines))
random.shuffle(combined)
labels, lines = zip(*combined)

max_pop_value = max([np.nanmax(pop_full) for pop_gaps, pop_full in lines])
y_limit = max(1500.0, max_pop_value * 1.05)

with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)
print("Saved ground truth to ground_truth.json")

# ==========================================
# PLOTTING THE TWO-PANEL GRAPH
# ==========================================

fig = plt.figure(figsize=(14, 15))
gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.15)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], polar=True)

start_vals = [pop_full[0] for pop_gaps, pop_full in lines]
sorted_indices = np.argsort(start_vals)

# --- Top Panel: Trajectories ---
for pop_gaps, pop_full in lines:
    plot_mask = t <= 100
    z = random.randint(10, 100)
    ax1.plot(t[plot_mask], pop_full[plot_mask], color='black', linewidth=1.2, solid_joinstyle='round', zorder=z)

text_y_positions = np.linspace(10, y_limit * 0.98, len(lines))

for rank, idx in enumerate(sorted_indices):
    actual_y = start_vals[idx]
    text_y = text_y_positions[rank]
    label = labels[idx]
    text_x = -15
    ax1.text(text_x, text_y, label, va='center', ha='right', fontsize=8, fontweight='bold', zorder=150)
    ax1.plot([text_x + 1, 0], [text_y, actual_y], color='gray', linewidth=0.8, zorder=150)

ax1.text(0.99, 0.96, "* Note: Species with >20% gaps are not valid.",
         transform=ax1.transAxes, ha='right', va='top', fontsize=11, color='black', fontstyle='italic',
         bbox=dict(facecolor='white', edgecolor='black', pad=4))

ax1.set_xlim(-35, 100)
ax1.set_ylim(0, y_limit)
ax1.set_ylabel("Population Size", fontsize=14, fontweight='bold', labelpad=10)
ax1.set_title("Population Dynamics of 35 Canadian Arctic Nature Reserve Introduced Species", fontsize=16,
              fontweight='bold', pad=20)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)

# --- Bottom Panel: Polar Data Availability Chart ---
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

ax2.set_ylim(0, 100)
# Simple grid, no colored bands or explicit 80% line
ax2.set_yticks([25, 50, 75, 100])
ax2.set_yticklabels(['25', '50', '75', '100'], color='gray', fontsize=8)
ax2.set_xticks(angles)
ax2.set_xticklabels(labels, fontsize=7, fontweight='bold')

for label, angle in zip(ax2.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

ax2.set_title("Survey of Data Captured", fontsize=12, fontweight='bold', pad=25)

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
            ax1.plot(t[dot_idx], pop_full[dot_idx], 'ro', markersize=6, zorder=200)

plt.savefig("population_dynamics_annotated.png", dpi=300, bbox_inches='tight')
print("Saved annotated two-panel visual to population_dynamics_annotated.png")

print("Visual generation complete. Exiting.")
sys.exit(0)