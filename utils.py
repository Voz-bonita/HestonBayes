import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def brl_formatter(value, tick_number):
    return f"R${int(value):,}".replace(",", ".")


def plot_wealth(dates, values, labels, title, path):
    fig, ax = plt.subplots(figsize=(10, 6))

    formatter = ticker.FuncFormatter(brl_formatter)
    ax.yaxis.set_major_formatter(formatter)
    for array, label in zip(values, labels):
        ax.plot(dates, array, label=label, marker="o")

    # Formatting the date axis
    fig.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
