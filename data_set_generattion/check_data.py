import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cases = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
dfs = {}

print("=== DELIVERABLE 3: KIRCHHOFF CURRENT LAW (KCL) CHECK ===")
print(f"{'Case':<12} | {'Rows':<6} | {'Mean |Id+Ig+Is+Ib| (A)':<24} | {'Max |Id+Ig+Is+Ib| (A)':<24}")
print("-" * 75)

for c in cases:
    df = pd.read_csv(f"{c}.csv")
    df['kcl_err'] = (df['id'] + df['ig'] + df['is'] + df['ib']).abs()
    dfs[c] = df
    print(f"{c:<12} | {len(df):<6} | {df['kcl_err'].mean():<24.4e} | {df['kcl_err'].max():<24.4e}")


def plot_binned_trend(ax, x, y, color, label):
    """Plots background scatter and overlays a 15-bin median trend curve."""
    ax.scatter(x, y, alpha=0.15, s=8, color=color)

    bins = np.linspace(x.min(), x.max(), 16)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    medians = [
        np.median(y[(x >= bins[i]) & (x < bins[i+1])])
        for i in range(len(bins)-1)
    ]

    ax.plot(
        bin_centers, medians,
        color=color, marker='o', linewidth=2.2,
        label=f"{label} (Median Trend)"
    )


# Filter OFF datasets for appreciable |Vds| > 0.1V
n_off = dfs["NMOS_OFF"][
    (dfs["NMOS_OFF"]['vd'] - dfs["NMOS_OFF"]['vs']).abs() > 0.1
]

p_off = dfs["PMOS_OFF"][
    (dfs["PMOS_OFF"]['vd'] - dfs["PMOS_OFF"]['vs']).abs() > 0.1
]


# ============================================================
# PLOT 1: OFF CURRENT vs TEMPERATURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

plot_binned_trend(ax, n_off['temp'].values, n_off['id'].abs().values,
                  'tab:blue', 'NMOS OFF |Id|')

plot_binned_trend(ax, p_off['temp'].values, p_off['id'].abs().values,
                  'tab:red', 'PMOS OFF |Id|')

ax.set_yscale('log')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('OFF Current |Id| (A) [Log Scale]')
ax.set_title('1. OFF Current vs. Temperature')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("plot1_off_current_vs_temperature.png", dpi=300)


# ============================================================
# PLOT 2: OFF CURRENT vs THRESHOLD VOLTAGE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

plot_binned_trend(ax, n_off['vth0'].abs().values, n_off['id'].abs().values,
                  'tab:blue', 'NMOS OFF |Id|')

plot_binned_trend(ax, p_off['vth0'].abs().values, p_off['id'].abs().values,
                  'tab:red', 'PMOS OFF |Id|')

ax.set_yscale('log')
ax.set_xlabel('Threshold Voltage Magnitude |vth0| (V)')
ax.set_ylabel('OFF Current |Id| (A) [Log Scale]')
ax.set_title('2. OFF Current vs. |vth0|')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("plot2_off_current_vs_vth0.png", dpi=300)


# ============================================================
# PLOT 3: GATE CURRENT vs OXIDE THICKNESS
# ============================================================

n_on = dfs["NMOS_ON"]
p_on = dfs["PMOS_ON"]

fig, ax = plt.subplots(figsize=(8, 5.5))

plot_binned_trend(ax, n_on['toxe'].values * 1e9, n_on['ig'].abs().values,
                  'tab:blue', 'NMOS ON |Ig|')

plot_binned_trend(ax, p_on['toxe'].values * 1e9, p_on['ig'].abs().values,
                  'tab:red', 'PMOS ON |Ig|')

ax.set_yscale('log')
ax.set_xlabel('Oxide Thickness toxe (nm)')
ax.set_ylabel('Gate Tunneling Current |Ig| (A) [Log Scale]')
ax.set_title('3. Gate Current |Ig| vs. toxe')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("plot3_gate_current_vs_toxe.png", dpi=300)


# ============================================================
# DISPLAY ALL THREE FIGURES
# ============================================================

print("\nSaved all three plots successfully:")
print("1. plot1_off_current_vs_temperature.png")
print("2. plot2_off_current_vs_vth0.png")
print("3. plot3_gate_current_vs_toxe.png")

plt.show()