import numpy as np
import matplotlib.pyplot as plt


def plot_model_metrics_radar(metrics, title="Model Performance Radar", save_path=None):
    """Plot a radar chart for model performance metrics.

    Parameters:
        metrics (dict): A mapping of model names to dicts of metric values.
            Example:
                {
                    "Logistic Regression": {"Accuracy": 0.90, "Precision": 0.91, ...},
                    "Random Forest": {"Accuracy": 0.97, "Precision": 0.97, ...},
                }
        title (str): Chart title.
        save_path (str): Optional file path to save the figure.
    """
    if not metrics:
        raise ValueError("metrics dictionary must not be empty")

    # Order metrics consistently across models
    metric_names = list(next(iter(metrics.values())).keys())
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], metric_names, color="black", size=11)
    ax.set_rlabel_position(180 / num_metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="gray", size=10)
    ax.grid(color="gray", linestyle="--", alpha=0.5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    for color, (model_name, model_metrics) in zip(colors, metrics.items()):
        values = [model_metrics[m] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, label=model_name, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_title(title, size=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    sample_metrics = {
        "Logistic Regression": {
            "Accuracy": 0.903142,
            "Specificity": 0.979522,
            "Precision": 0.909469,
            "Sensitivity": 0.903142,
            "F1 Score": 0.901832,
        },
        "Random Forest": {
            "Accuracy": 0.972156,
            "Specificity": 0.997696,
            "Precision": 0.972169,
            "Sensitivity": 0.972156,
            "F1 Score": 0.972093,
        },
        "SVC": {
            "Accuracy": 0.517303,
            "Specificity": 0.452555,
            "Precision": 0.531032,
            "Sensitivity": 0.517303,
            "F1 Score": 0.450838,
        },
        "KNN": {
            "Accuracy": 0.770286,
            "Specificity": 0.741313,
            "Precision": 0.771799,
            "Sensitivity": 0.770286,
            "F1 Score": 0.768489,
        },
        "Decision Tree": {
            "Accuracy": 0.953461,
            "Specificity": 0.997596,
            "Precision": 0.953299,
            "Sensitivity": 0.953461,
            "F1 Score": 0.953354,
        },
    }

    plot_model_metrics_radar(sample_metrics, title="Model Performance Radar Plot")
