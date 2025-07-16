import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
import re
import csv
import pandas as pd
from tqdm import tqdm

# Configurar GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Función zeta_RD refinada
def zeta_RD(t_range, t_n_sub, batch_size=1000, use_abs=False, normalize=True):
    t_range = torch.tensor(t_range, device=device)
    t_n_sub = torch.tensor(t_n_sub, device=device)
    result = []
    for i in range(0, len(t_range), batch_size):
        t_batch = t_range[i:i+batch_size]
        arg = torch.pi * t_batch[:, None] / t_n_sub[None, :]
        sin_val = torch.sin(arg)
        sin_val = torch.clamp(sin_val, -1, 1)
        x = sin_val
        x = x - torch.floor(x)
        val = (9 * x).int()
        rd9_val = 1 + (val - 1) % 9
        rd9_val[val == 0] = 9
        sum_val = torch.sum(rd9_val, dim=1)
        if normalize:
            sum_val = sum_val / t_n_sub.shape[0]
        result.append(sum_val)
    return torch.cat(result).cpu().numpy()

# Validación con refinamiento parabólico
def validate_minimum_direct(t_k, t_n_sub, range_width=0.1, num_points=50000):
    torch.cuda.empty_cache()
    t_range = np.linspace(t_k - range_width, t_k + range_width, num_points)
    zeta_vals = zeta_RD(t_range, t_n_sub)

    min_indices, _ = find_peaks(-zeta_vals)
    if len(min_indices) == 0:
        return None

    min_idx = min_indices[np.argmin(zeta_vals[min_indices])]

    if 1 < min_idx < len(zeta_vals) - 2:
        x1 = t_range[min_idx - 1]
        x2 = t_range[min_idx]
        x3 = t_range[min_idx + 1]
        y1 = zeta_vals[min_idx - 1]
        y2 = zeta_vals[min_idx]
        y3 = zeta_vals[min_idx + 1]

        denom = (y1 - 2*y2 + y3)
        if denom != 0:
            t_refined = x2 - 0.5 * ((y3 - y1) / denom) * (x3 - x1) / 2
        else:
            t_refined = x2
    else:
        t_refined = t_range[min_idx]

    return abs(t_refined - t_k)

# Cargar ceros bajos (M = 2M)
t_n_list = np.loadtxt("odlyzko_zeros.txt", skiprows=0)
M = 2_000_000
t_n_sub = t_n_list[:M]
print(f"Cargados {M} ceros bajos.")

# Cargar ceros altos
with open("ceros_10^22.rtf", "r") as file:
    raw_lines = file.readlines()

gamma_vals = []
for line in raw_lines:
    matches = re.findall(r"\d+\.\d+", line.replace(",", "."))
    for match in matches:
        gamma_vals.append(float(match))

gamma_vals = np.array(gamma_vals)
print(f"Cargados {len(gamma_vals)} ceros altos.")

# Validar últimos N ceros
N = 100
errores = []
indices = []
gamas = []

print(f"\nValidando {N} ceros altos con refinamiento parabólico...")
for i, gamma in enumerate(tqdm(gamma_vals[-N:])):
    error = validate_minimum_direct(gamma, t_n_sub)
    if error is not None:
        errores.append(error)
        indices.append(len(gamma_vals) - N + i + 1)
        gamas.append(gamma)

# Guardar CSV
csv_name = "errores_ceros_10^22_RD9_refinada_parabola.csv"
with open(csv_name, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gamma_index", "gamma_value", "error"])
    for idx, gamma, error in zip(indices, gamas, errores):
        writer.writerow([idx, gamma, error])
print(f"✅ Resultados guardados en {csv_name}")

# Resumen
df = pd.DataFrame({"gamma_index": indices, "gamma_value": gamas, "error": errores})
min_error = df["error"].min()
max_error = df["error"].max()
mean_error = df["error"].mean()
under_threshold = (df["error"] < 0.1).sum()
total = len(df)

print(f"\n📊 Resumen refinado con parábola:")
print(f"🔸 Error mínimo:  {min_error:.6f}")
print(f"🔸 Error máximo:  {max_error:.6f}")
print(f"🔸 Error promedio: {mean_error:.6f}")
print(f"✅ {under_threshold}/{total} ({under_threshold/total*100:.1f}%) con error < 0.1")

# Visualización
plt.figure(figsize=(10, 5))
plt.scatter(df["gamma_value"], df["error"], s=10, color="navy", alpha=0.7)
plt.axhline(y=0.1, color='green', linestyle='--', label='Umbral deseado (0.1)')
plt.axhline(y=0.5, color='red', linestyle='--', label='Límite aceptable (0.5)')
plt.xlabel("γ (Altura del cero)")
plt.ylabel("Error absoluto")
plt.title("Precisión con Refinamiento Parabólico en Ceros ~10^22")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("grafico_zeta_rd_refinada_parabola.png", dpi=300)
plt.show()
