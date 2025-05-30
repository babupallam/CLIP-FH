import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
vitb_csv = "result_logs/stage3_vitb16_train_table.csv"
rn50_csv = "result_logs/stage3_rn50_train_table.csv"
output_dir = "result_logs"
os.makedirs(output_dir, exist_ok=True)

# Load data
df_vitb = pd.read_csv(vitb_csv)
df_rn50 = pd.read_csv(rn50_csv)

sns.set(style="whitegrid")

def plot_metric_vs_epoch(df, model, prefix, metric, ylabel, suffix):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y=metric, hue="version", marker="o")
    # annotate best points
    for version, group in df.groupby("version"):
        best_row = group.loc[group[metric].idxmax()]
        plt.scatter(best_row["epoch"], best_row[metric], s=40, color="black")
        plt.text(best_row["epoch"], best_row[metric] + 0.3,
                 f"{best_row[metric]:.1f}", fontsize=8, ha="center")
    plt.title(f"{model.upper()} - {ylabel} vs Epoch")
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend(title="Version", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{suffix}.png"))
    plt.close()

def plot_final_metric_per_version(df, model, prefix, metric, ylabel, suffix):
    final_df = df.sort_values("epoch").groupby("version").last()
    final_df = final_df.sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=final_df.index, y=final_df[metric], hue=final_df.index, palette="Blues_d", legend=False)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                f'{height:.1f}', ha="center", fontsize=8)
    plt.title(f"{model.upper()} - Final {ylabel} per Version")
    plt.ylabel(ylabel)
    plt.xlabel("Version")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{suffix}.png"))
    plt.close()



# Run for both models
for model, df, prefix in [("vitb16", df_vitb, os.path.splitext(os.path.basename(vitb_csv))[0]),
                          ("rn50", df_rn50, os.path.splitext(os.path.basename(rn50_csv))[0])]:

    # Rank-1 / mAP vs Epoch
    plot_metric_vs_epoch(df, model, prefix, "val_rank1", "Rank-1 Accuracy", "val_rank1_vs_epoch")
    plot_metric_vs_epoch(df, model, prefix, "val_mAP", "Mean Average Precision", "val_map_vs_epoch")

    # Final metrics (bar chart)
    plot_final_metric_per_version(df, model, prefix, "val_rank1", "Rank-1 Accuracy", "final_val_rank1")
    plot_final_metric_per_version(df, model, prefix, "val_mAP", "Mean Average Precision", "final_val_map")


print("Stage 3 plots saved to result_logs/")
