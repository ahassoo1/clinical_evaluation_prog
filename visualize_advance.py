import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
from sklearn.metrics import cohen_kappa_score

OUTPUT_FOLDER = 'output'
PNG_FILE_PATH = os.path.join(OUTPUT_FOLDER, 'llm_vs_human_diverging_bar_chart.png')
PDF_FILE_PATH = os.path.join(OUTPUT_FOLDER, 'llm_vs_human_diverging_bar_chart.pdf')


def find_latest_evaluation_file(folder: str) -> str | None:
    search_path = os.path.join(folder, 'dialogue_evaluations_human_*.csv')
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def main():
    print("--- Running Script: Generate Professional Diverging Bar Chart ---")

    latest_eval_file = find_latest_evaluation_file(OUTPUT_FOLDER)
    if not latest_eval_file:
        print(f"FATAL: No evaluation CSV files found in the '{OUTPUT_FOLDER}' directory.")
        return

    print(f"Loading data from: {os.path.basename(latest_eval_file)}")
    try:
        df = pd.read_csv(latest_eval_file)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 15,
        'figure.titlesize': 24
    })

    llm_score_cols = [
        col for col in df.columns
        if col.endswith('_Score') and 'Human' not in col and 'Synthetic' not in col
    ]
    criteria_names = [col.replace('_Score', '') for col in llm_score_cols]
    human_score_cols = [f"{name}_Human_Score" for name in criteria_names]
    if not all(col in df.columns for col in human_score_cols):
        print("\nFATAL: One or more human score columns are missing.")
        return
    plot_df = df.dropna(subset=llm_score_cols + human_score_cols).copy()
    if plot_df.empty:
        print("\nFATAL: No valid rows for comparison after removing missing scores.")
        return

    # Calculate counts and percentages
    total_per_criterion = {name: plot_df[hc].notna().sum() for name, hc in zip(criteria_names, human_score_cols)}
    llm_counts = {col.replace('_Score', ''): plot_df[col].value_counts(normalize=True) for col in llm_score_cols}
    llm_counts_df = pd.DataFrame(llm_counts).T.fillna(0)
    human_counts = {col.replace('_Human_Score', ''):
                    plot_df[col].value_counts(normalize=True) for col in human_score_cols}
    human_counts_df = pd.DataFrame(human_counts).T.fillna(0)
    for score in [1, 3, 5]:
        if score not in llm_counts_df.columns: llm_counts_df[score] = 0
        if score not in human_counts_df.columns: human_counts_df[score] = 0
    llm_counts_df = llm_counts_df.reindex(criteria_names)
    human_counts_df = human_counts_df.reindex(criteria_names)

    # Calculate Cohen's kappa per criterion
    kappas = [
        cohen_kappa_score(plot_df[hc], plot_df[lc])
        for hc, lc in zip(human_score_cols, llm_score_cols)
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 10))
    score_colors = {1: "#a1dab4", 3: "#41b6c4", 5: "#225ea8"}
    score_labels = {1: "1 (Major Risk)", 3: "3 (Potential Risk)", 5: "5 (No Risk)"}
    y_pos = np.arange(len(criteria_names))
    left_human = np.zeros(len(criteria_names))
    left_llm = np.zeros(len(criteria_names))

    for score in [1, 3, 5]:
        human_vals = human_counts_df[score].values * 100  # percent
        llm_vals = llm_counts_df[score].values * 100
        ax.barh(y_pos, human_vals, left=left_human, color=score_colors[score],
                label=score_labels[score], align='center')
        ax.barh(y_pos, -llm_vals, left=-left_llm, color=score_colors[score], align='center')
        for i, (hv, lv) in enumerate(zip(human_vals, llm_vals)):
            if hv > 0:
                ax.text(left_human[i] + hv / 2, i, f"{hv:.1f}%", ha='center', va='center', color='black', fontsize=13, weight='bold')
            if lv > 0:
                ax.text(-left_llm[i] - lv / 2, i, f"{lv:.1f}%", ha='center', va='center', color='black', fontsize=13, weight='bold')
        left_human += human_vals
        left_llm += llm_vals

    ax.set_yticks(y_pos)
    # Annotate with kappa
    new_labels = [
        f"{name.replace(' Quality', '')}\n(Kappa={kappas[i]:.2f}, N={total_per_criterion[name]})"
        for i, name in enumerate(criteria_names)
    ]
    ax.set_yticklabels(new_labels, fontsize=15)
    ax.invert_yaxis()
    max_val = max(left_human.max(), left_llm.max())
    ax.set_xlim(-max_val * 1.15, max_val * 1.15)
    ticks = np.linspace(-max_val, max_val, 9)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{abs(int(x))}%' for x in ax.get_xticks()])
    ax.set_xlabel("Percentage of Dialogues", fontsize=16)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.text(0.22, 1.04, "LLM Scores", transform=ax.transAxes, ha='center', fontsize=18, weight='bold')
    ax.text(0.78, 1.04, "Human Scores", transform=ax.transAxes, ha='center', fontsize=18, weight='bold')
    fig.suptitle("LLM vs. Human Score Distribution by Criterion", fontsize=25, weight='bold')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Score", bbox_to_anchor=(0.5, -0.13), loc='upper center', ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(PNG_FILE_PATH, dpi=320)
    plt.savefig(PDF_FILE_PATH, bbox_inches='tight')
    print(f"\n-> Diverging bar chart saved to '{os.path.abspath(PNG_FILE_PATH)}' and PDF.")
    print("\nVisualization complete.")


if __name__ == '__main__':
    main()
