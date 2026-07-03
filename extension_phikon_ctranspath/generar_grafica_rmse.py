"""
generar_grafica_rmse.py
Genera la gráfica de evolución del RMSE de validación a partir de un log
real de entrenamiento, sin copiar ni pegar valores a mano. Así la gráfica
es 100% reproducible y trazable hasta el fichero de log original.

Uso:
    python generar_grafica_rmse.py <fichero_log> <titulo_extra> <fichero_salida>

Ejemplos:
    python generar_grafica_rmse.py os_time_train_regularizado_log.txt \
        "Dual_InceptionV3 con regularización L2, weight_decay=1e-4 (baseline)" \
        grafica_rmse_baseline_regularizado.png

    python generar_grafica_rmse.py os_time_train_phikon_log.txt \
        "Dual_InceptionV3 con regularización L2 (cluster 05, Phikon-v2)" \
        grafica_rmse_phikon_cluster05.png
"""
import re
import sys
import matplotlib.pyplot as plt


def extraer_val_rmse(log_path):
    """Lee el log línea a línea y extrae el val_rmse de cada época,
    en el mismo orden en que aparecen en el fichero."""
    val_rmse = []
    patron = re.compile(r"val_rmse:\s*([0-9.]+)")
    with open(log_path, "r") as f:
        for linea in f:
            match = patron.search(linea)
            if match:
                val_rmse.append(float(match.group(1)))
    return val_rmse


def main():
    if len(sys.argv) < 4:
        log_path = "os_time_train_regularizado_log.txt"
        titulo_extra = "Dual_InceptionV3 con regularización L2, weight_decay=1e-4"
        output_path = "grafica_rmse_baseline_regularizado.png"
    else:
        log_path = sys.argv[1]
        titulo_extra = sys.argv[2]
        output_path = sys.argv[3]

    val_rmse = extraer_val_rmse(log_path)
    if not val_rmse:
        raise RuntimeError(
            f"No se encontró ningún valor de val_rmse en {log_path}. "
            "Comprueba que el fichero existe y tiene el formato esperado."
        )

    epocas = list(range(1, len(val_rmse) + 1))
    min_val = min(val_rmse)
    min_epoca = val_rmse.index(min_val) + 1

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(epocas, val_rmse, color="#2563eb", linewidth=1.4, alpha=0.85,
            label="RMSE de validación")

    ax.scatter([min_epoca], [min_val], color="#dc2626", zorder=5, s=70,
               label=f"Mínimo: {min_val:.1f} días (época {min_epoca})")
    ax.annotate(f"{min_val:.1f}", (min_epoca, min_val),
                textcoords="offset points", xytext=(10, -15), fontsize=10,
                color="#dc2626", fontweight="bold")

    ax.axhline(y=min_val, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("RMSE de validación (días)", fontsize=11)
    ax.set_title(
        f"Evolución del RMSE de validación durante el entrenamiento\n({titulo_extra})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlim(0, len(val_rmse) + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Gráfica guardada en: {output_path}")
    print(f"Total de épocas leídas del log: {len(val_rmse)}")
    print(f"Mínimo: {min_val:.3f} días, en la época {min_epoca}")


if __name__ == "__main__":
    main()
