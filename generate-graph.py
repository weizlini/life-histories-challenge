import numpy as np
import matplotlib.pyplot as plt
import json
import random
import sys

# Generate high-resolution time data
t = np.linspace(0, 100, 3000)

# Seed for reproducibility so the graph and JSON match every time
np.random.seed(42)
random.seed(42)

lines = []
labels = []
ground_truth = {"species_count": 35, "k_selected_count": 10, "r_selected_count": 25, "classifications": {}}

# Guarantee unique, completely mixed starting values for all 35 species
# by pulling them all from a single combined pool
all_target_starts = random.sample(range(20, 950), 35)
k_starts = all_target_starts[:10]
r_starts = all_target_starts[10:]

# ==========================================
# POPULATION MATH (35 Species)
# ==========================================

# Generate 10 K-selected species (Logistic growth/decline or stable carrying capacity)
for i in range(10):
    name = f"Species {i + 1}"
    labels.append(name)
    ground_truth["classifications"][name] = "K-selected"

    start_pop = k_starts[i]
    end_pop = np.random.randint(100, 900)
    midpoint = np.random.randint(20, 80)
    rate = np.random.uniform(0.1, 0.3)

    # Base logistic curve
    pop = start_pop + (end_pop - start_pop) / (1 + np.exp(-rate * (t - midpoint)))
    # Add minor environmental wobble typical of K-selected hovering at carrying capacity
    pop += np.random.randint(5, 25) * np.sin(t * np.random.uniform(0.05, 0.2))

    lines.append(pop)

# Generate 25 r-selected species (Boom and bust exponential spikes)
for i in range(25):
    name = f"Species {i + 11}"
    labels.append(name)
    ground_truth["classifications"][name] = "r-selected"

    target_start = float(r_starts[i])

    # If the randomly assigned start value is high, model a "pre-crash" state at t=0
    if target_start > 250:
        base_pop = float(np.random.randint(10, 150))
        pop = np.full_like(t, base_pop, dtype=float)

        # Center a spike at t=0 so the actual starting point matches target_start, then crashes
        spike_height = target_start - base_pop
        width = np.random.uniform(2, 8)
        pop += spike_height * np.exp(-((t - 0) ** 2) / width)
    else:
        # Standard low starting baseline
        base_pop = target_start
        pop = np.full_like(t, base_pop, dtype=float)

    # 1 to 4 massive exponential reproduction events in the future
    num_spikes = np.random.randint(1, 5)
    for _ in range(num_spikes):
        spike_time = np.random.randint(10, 90)
        spike_height = np.random.randint(400, 900)
        width = np.random.uniform(2, 8)  # narrowness of the spike
        pop += spike_height * np.exp(-((t - spike_time) ** 2) / width)

    lines.append(pop)

# Shuffle the plotting order so they aren't visually layered strictly by type
combined = list(zip(labels, lines))
random.shuffle(combined)
labels, lines = zip(*combined)

# ==========================================
# EXPORT GROUND TRUTH JSON
# ==========================================
with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)
print("Saved ground truth to ground_truth.json")

# ==========================================
# PLOTTING THE GRAPH
# ==========================================

fig, ax = plt.subplots(figsize=(14, 10))

for pop in lines:
    ax.plot(t, pop, color='black', linewidth=1.0, solid_joinstyle='round', alpha=0.8)

# Calculate start values to organize labels without overlapping lines
start_vals = [pop[0] for pop in lines]
sorted_indices = np.argsort(start_vals)

# Create 35 evenly spaced y-coordinates for our text labels
text_y_positions = np.linspace(10, 1090, len(lines))

for rank, idx in enumerate(sorted_indices):
    actual_y = start_vals[idx]
    text_y = text_y_positions[rank]
    label = labels[idx]

    text_x = -15  # Pull the text back into the left margin

    # Add the text label
    ax.text(text_x, text_y, label, va='center', ha='right',
            fontsize=8, fontweight='bold', fontfamily='sans-serif')

    # Draw a leader line connecting the text space to the starting population point
    ax.plot([text_x + 1, 0], [text_y, actual_y], color='black', linewidth=0.5, alpha=0.6)

# Expand the left margin to -35 to give plenty of room for labels and leader lines
ax.set_xlim(-35, 105)
ax.set_ylim(0, 1100)
ax.set_xlabel("Time (Generations)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Population Size", fontsize=14, fontweight='bold', labelpad=10)
ax.set_title("Population Dynamics of 35 Unidentified Species", fontsize=16, fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.grid(False)

plt.tight_layout()
plt.show()

# Terminate script
print("Visual generation complete. Exiting.")
sys.exit(0)