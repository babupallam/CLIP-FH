import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
vitb_csv = "result_logs/stage2_vitb16_train_table.csv"
rn50_csv = "result_logs/stage2_rn50_train_table.csv"
output_dir = "result_logs"
os.makedirs(output_dir, exist_ok=True)

# Load data
df_vitb = pd.read_csv(vitb_csv)
df_rn50 = pd.read_csv(rn50_csv)

sns.set(style="whitegrid")

def plot_metric_vs_epoch(df, model, prefix, metric, ylabel, suffix):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y=metric, hue="version", marker="o")

    # Annotate best point per version
    for version, group in df.groupby("version"):
        best_row = group.loc[group[metric].idxmax()]
        plt.scatter(best_row["epoch"], best_row[metric], s=50, color="black")
        plt.text(best_row["epoch"], best_row[metric] + 0.5,
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

    # Add value labels on bars
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

def plot_loss_facet(df, model, prefix):
    loss_cols = ["loss_id", "loss_triplet", "loss_center", "loss_i2t", "loss_t2i"]
    loss_df = df[["epoch", "version"] + loss_cols].copy()
    melted = pd.melt(loss_df, id_vars=["epoch", "version"], value_vars=loss_cols,
                     var_name="loss_type", value_name="loss_value")

    g = sns.relplot(
        data=melted,
        x="epoch", y="loss_value",
        hue="version", col="loss_type",
        kind="line", col_wrap=2,
        facet_kws={'sharey': False, 'sharex': True},
        height=4, aspect=1.5
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{model.upper()} - Loss Breakdown per Epoch")
    g.savefig(os.path.join(output_dir, f"{prefix}_loss_breakdown_facet.png"))
    plt.close()


def plot_train_loss_vs_epoch(df, model, prefix):
    # Ensure all required loss columns exist
    loss_cols = ["loss_id", "loss_triplet", "loss_center", "loss_i2t", "loss_t2i"]
    missing = [col for col in loss_cols if col not in df.columns]
    if missing:
        print(f"[!] Skipping loss plot for {model} (missing: {missing})")
        return

    # Compute total loss
    df = df.copy()
    df["total_loss"] = df[loss_cols].sum(axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="total_loss", hue="version", marker="o")
    plt.title(f"{model.upper()} - Total Training Loss vs Epoch")
    plt.ylabel("Total Training Loss")
    plt.xlabel("Epoch")
    plt.legend(title="Version", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_loss_vs_epoch.png"))
    plt.close()


# Run for each model
for model, df, prefix in [("vitb16", df_vitb, os.path.splitext(os.path.basename(vitb_csv))[0]),
                          ("rn50", df_rn50, os.path.splitext(os.path.basename(rn50_csv))[0])]:

    # Rank-1 and mAP vs Epoch (with best epoch marker)
    plot_metric_vs_epoch(df, model, prefix, "rank1", "Rank-1 Accuracy", "rank1_vs_epoch")
    plot_metric_vs_epoch(df, model, prefix, "map", "Mean Average Precision", "map_vs_epoch")

    # Sorted bar plots of final performance with value labels
    plot_final_metric_per_version(df, model, prefix, "rank1", "Rank-1 Accuracy", "final_rank1_per_version")
    plot_final_metric_per_version(df, model, prefix, "map", "Mean Average Precision", "final_map_per_version")

    # Faceted loss breakdown plot
    plot_loss_facet(df, model, prefix)

    # Training loss vs epoch
    plot_train_loss_vs_epoch(df, model, prefix)

print("Stage 2 plots saved to result_logs/")
