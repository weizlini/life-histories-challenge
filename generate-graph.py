import numpy as np
import matplotlib.pyplot as plt
import json
import random
import sys

# Generate high-resolution time data
t = np.linspace(0, 110, 3000)

# Seed for reproducibility so the graph and JSON match every time
np.random.seed(42)
random.seed(42)

lines = []
labels = []
ground_truth = {"species_count": 35, "k_selected_count": 10, "r_selected_count": 25, "classifications": {}}

# Guarantee unique, completely mixed starting values for all 35 species
all_target_starts = random.sample(range(20, 950), 35)
k_starts = all_target_starts[:10]
r_starts = all_target_starts[10:]


def inject_missing_data(pop_array, time_array):
    """Simulates missed ecological surveys by injecting np.nan gaps."""
    num_gaps = np.random.randint(2, 6)
    for _ in range(num_gaps):
        gap_start = np.random.uniform(5, 95)
        gap_len = np.random.uniform(0.5, 2.5)  # Gap of 0.5 to 2.5 years
        gap_mask = (time_array >= gap_start) & (time_array <= gap_start + gap_len)
        pop_array[gap_mask] = np.nan
    return pop_array


# ==========================================
# POPULATION MATH (35 Species)
# ==========================================

for i in range(10):
    name = f"Species {i + 1}"
    labels.append(name)
    ground_truth["classifications"][name] = "K-selected"

    start_pop = k_starts[i]
    end_pop = np.random.randint(100, 900)
    midpoint = np.random.randint(20, 80)
    rate = np.random.uniform(0.05, 0.2)

    # 1. Base logistic curve
    pop = start_pop + (end_pop - start_pop) / (1 + np.exp(-rate * (t - midpoint)))

    # 2. Add VERTICAL crashes to roughly half the K-selected species to trick derivative filters
    if random.random() > 0.5:
        crash_time = np.random.uniform(20, 80)
        crash_idx = int((crash_time / 110.0) * 3000)
        if crash_idx < 3000:
            current_level = pop[crash_idx]
            crash_depth = np.random.uniform(0.4, 0.7) * current_level
            # HARDENING: Massive steepness ensures the drop is nearly vertical (dy/dx > 5.0)
            crash_steepness = np.random.uniform(5.0, 15.0)

            pop -= crash_depth / (1 + np.exp(-crash_steepness * (t - crash_time)))

            recovery_rate = np.random.uniform(0.02, 0.08)
            pop += (crash_depth * np.random.uniform(0.3, 0.8)) / (1 + np.exp(-recovery_rate * (t - (crash_time + 15))))

    # 3. Add SHATTER Jitter: high frequency/amplitude seasonal wobble.
    # Mathematically creates a local slope > 3.5 continuously, shattering the line in a slope filter.
    pop += np.random.uniform(15, 35) * np.sin(t * np.random.uniform(3.0, 6.0))
    pop += np.random.randint(2, 10) * np.cos(t * np.random.uniform(0.8, 2.0))

    pop = np.clip(pop, 5, None)
    pop = inject_missing_data(pop, t)
    lines.append(pop)

for i in range(25):
    name = f"Species {i + 11}"
    labels.append(name)
    ground_truth["classifications"][name] = "r-selected"

    target_start = float(r_starts[i])

    if target_start > 250:
        # Delayed crash to baseline
        base_pop = np.random.uniform(10.0, 180.0)
        crash_midpoint = np.random.uniform(5, 25)
        crash_rate = np.random.uniform(0.2, 0.5)

        height = (target_start - base_pop) * (1 + np.exp(-crash_rate * crash_midpoint))
        pop = base_pop + height / (1 + np.exp(crash_rate * (t - crash_midpoint)))
        spike_start = 30
    else:
        if random.random() > 0.5:
            # The "Lag Phase" Trap (Fake K-selected)
            fake_k_end = np.random.uniform(target_start + 50, target_start + 300)
            fake_mid = np.random.uniform(20, 50)
            fake_rate = np.random.uniform(0.05, 0.15)

            pop = target_start + (fake_k_end - target_start) / (1 + np.exp(-fake_rate * (t - fake_mid)))
            spike_start = np.random.uniform(55, 75)
        else:
            # Standard low starting baseline with an early boom
            base_pop = target_start
            pop = np.full_like(t, base_pop, dtype=float)

            early_spike = np.random.uniform(5, 15)
            early_height = np.random.randint(400, 900)
            early_width = np.random.uniform(2, 6)
            pop += early_height * np.exp(-((t - early_spike) ** 2) / early_width)
            spike_start = 30

    pop += np.random.uniform(2, 8) * np.sin(t * np.random.uniform(0.01, 0.1))

    # Add the exponential spikes
    num_spikes = np.random.randint(2, 6)
    for _ in range(num_spikes):
        spike_time = np.random.uniform(spike_start, 110)
        spike_height = np.random.randint(400, 900)

        # HARDENING: 30% chance for a "Wide Hill" boom that stays below AI's slope threshold
        if random.random() > 0.7:
            width = np.random.uniform(30, 80)
        else:
            width = np.random.uniform(2, 8)

        pop += spike_height * np.exp(-((t - spike_time) ** 2) / width)

    pop = np.clip(pop, 5, None)
    pop = inject_missing_data(pop, t)
    lines.append(pop)

combined = list(zip(labels, lines))
random.shuffle(combined)
labels, lines = zip(*combined)

# Find the absolute highest point across all species to dynamically set the Y-axis limit
# We use np.nanmax because the arrays now contain NaNs
max_pop_value = max([np.nanmax(pop) for pop in lines])
y_limit = max(1100.0, max_pop_value * 1.05)

# ==========================================
# EXPORT GROUND TRUTH JSON
# ==========================================
with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)
print("Saved ground truth to ground_truth.json")

# ==========================================
# PLOTTING THE CLEAN GRAPH
# ==========================================

fig, ax = plt.subplots(figsize=(14, 10))

for pop in lines:
    plot_mask = t <= 100
    ax.plot(t[plot_mask], pop[plot_mask], color='black', linewidth=1.0, solid_joinstyle='round', alpha=0.8)

# Get valid starting values (t=0) despite random initial nan gaps
start_vals = []
for pop in lines:
    first_valid_idx = np.where(~np.isnan(pop))[0][0]
    start_vals.append(pop[first_valid_idx])

sorted_indices = np.argsort(start_vals)
text_y_positions = np.linspace(10, y_limit * 0.98, len(lines))

for rank, idx in enumerate(sorted_indices):
    actual_y = start_vals[idx]
    text_y = text_y_positions[rank]
    label = labels[idx]

    text_x = -15
    ax.text(text_x, text_y, label, va='center', ha='right',
            fontsize=8, fontweight='bold', fontfamily='sans-serif')
    ax.plot([text_x + 1, 0], [text_y, actual_y], color='black', linewidth=0.5, alpha=0.6)

ax.set_xlim(-35, 100)
ax.set_ylim(0, y_limit)
ax.set_xlabel("Time (Years)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Population Size", fontsize=14, fontweight='bold', labelpad=10)
ax.set_title("Population Dynamics of 35 Canadian Arctic Nature Reserve Introduced species", fontsize=16,
             fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.grid(False)

plt.tight_layout()

plt.savefig("population_dynamics.png", dpi=300, bbox_inches='tight')
print("Saved clean visual to population_dynamics.png")

# ==========================================
# PLOTTING THE ANNOTATED GRAPH
# ==========================================

for i, pop in enumerate(lines):
    label = labels[i]
    if ground_truth["classifications"][label] == "K-selected":
        plot_mask = t <= 100
        visible_indices = np.where(plot_mask & ~np.isnan(pop))[0]
        if len(visible_indices) > 0:
            dot_idx = random.choice(visible_indices[500:-500])
            ax.plot(t[dot_idx], pop[dot_idx], 'ro', markersize=6, zorder=5)

plt.savefig("population_dynamics_annotated.png", dpi=300, bbox_inches='tight')
print("Saved annotated visual to population_dynamics_annotated.png")

print("Visual generation complete. Exiting.")
sys.exit(0)