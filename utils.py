import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_wealth(dates, values, labels, title, path):
    plt.figure(figsize=(10, 6))
    for array, label in zip(values, labels):
        plt.plot(dates, array, label, marker="o")

    # Formatting the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
