import matplotlib.pyplot as plt

def plot_equity(curves: dict, out_path: str):
    plt.figure(figsize=(10, 5))
    for label, series in curves.items():
        series.plot(label=label)
    plt.legend()
    plt.title("Equity Curves")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
