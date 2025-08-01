import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# --- CONFIGURATION ---
OUTPUT_FOLDER = 'output'
PNG_FILE_PATH = os.path.join(OUTPUT_FOLDER, 'human_vs_llm_comparison.png')
PDF_FILE_PATH = os.path.join(OUTPUT_FOLDER, 'human_vs_llm_comparison.pdf')

# --- SETUP MATPLOTLIB/SEABORN STYLE ---
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 15,
    'axes.titlesize': 22,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'figure.titlesize': 26
})


def find_latest_evaluation_file(folder: str) -> str | None:
    """Finds the most recent CSV file with human scores."""
    search_path = os.path.join(folder, 'dialogue_evaluations_human_*.csv')
    files = glob.glob(search_path)
    if not files:
        search_path = os.path.join(folder, 'dialogue_evaluations_*.csv')
        files = glob.glob(search_path)
        if not files:
            return None
    return max(files, key=os.path.getctime)


def main():
    print("=== Human vs. LLM Evaluation Visualization ===")
    latest_eval_file = find_latest_evaluation_file(OUTPUT_FOLDER)
    if not latest_eval_file:
        print("No evaluation CSV files found in 'output'.")
        return

    print(f"Loading data: {os.path.basename(latest_eval_file)}")
    df = pd.read_csv(latest_eval_file)

    # --- Identify score columns ---
    llm_score_cols = [c for c in df.columns if c.endswith('_Score') and 'Human' not in c and 'Synthetic' not in c]
    criteria_names = [col.replace('_Score', '') for col in llm_score_cols]
    human_score_cols = [f"{name}_Human_Score" for name in criteria_names]

    if not all(col in df.columns for col in human_score_cols):
        print("Missing required human score columns.")
        return

    plot_df = df.dropna(subset=llm_score_cols + human_score_cols).copy()
    if plot_df.empty:
        print("No valid rows after dropping missing data.")
        return

    # --- Create figure with subplots: [bar, confusion, scatter] for each criterion ---
    n_criteria = len(criteria_names)
    fig, axes = plt.subplots(n_criteria, 3, figsize=(20, 6 * n_criteria))
    if n_criteria == 1:  # For 1 criterion, axes are 1D
        axes = axes.reshape(1, 3)

    fig.suptitle("Human vs. LLM Clinical Dialogue Evaluation", weight='bold', y=0.995)

    for i, name in enumerate(criteria_names):
        llm_col = llm_score_cols[i]
        human_col = human_score_cols[i]
        llm_scores = plot_df[llm_col]
        human_scores = plot_df[human_col]
        total = len(plot_df)

        # Agreement statistics
        raw_agreement = (llm_scores == human_scores).mean() * 100
        kappa = cohen_kappa_score(human_scores, llm_scores)
        agreement_str = f"Raw Agreement: {raw_agreement:.1f}%\nCohen's Kappa: {kappa:.2f}\nN={total}"

        # --- Barplot: Score Distribution ---
        ax0 = axes[i, 0]
        melted = plot_df[[llm_col, human_col]].melt(var_name='Rater', value_name='Score')
        melted['Rater'] = melted['Rater'].map({llm_col: "LLM", human_col: "Human"})
        sns.countplot(
            data=melted, x='Score', hue='Rater', order=[1, 3, 5],
            palette=['#377eb8', '#e41a1c'], ax=ax0
        )
        ax0.set_title(f"{name}\nScore Distribution", pad=18)
        ax0.set_xlabel("Score")
        ax0.set_ylabel("Count")
        # Annotate bar percentages
        for p in ax0.patches:
            if p.get_height() > 0:
                percent = 100 * p.get_height() / total
                ax0.annotate(f"{int(p.get_height())}\n({percent:.1f}%)",
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=11,
                             xytext=(0, 8), textcoords='offset points')

        # --- Confusion Matrix (normalized) ---
        ax1 = axes[i, 1]
        cm = confusion_matrix(human_scores, llm_scores, labels=[1, 3, 5], normalize='true')
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', ax=ax1,
                    cbar=False, square=True,
                    xticklabels=[1, 3, 5], yticklabels=[1, 3, 5],
                    annot_kws={"size": 14, "weight": "bold"})
        ax1.set_title(f"{name}\nNormalized Confusion Matrix", pad=18)
        ax1.set_xlabel("LLM Predicted Score")
        ax1.set_ylabel("Human (True) Score")

        # --- Scatter Plot: Score Agreement ---
        ax2 = axes[i, 2]
        sns.stripplot(x=human_scores, y=llm_scores, ax=ax2,
                      order=[1, 3, 5], color='#555', jitter=0.2, size=7, alpha=0.6)
        ax2.plot([1, 5], [1, 5], ls='--', color='gray', lw=1)
        ax2.set_title(f"{name}\nScore Agreement (Scatter)", pad=18)
        ax2.set_xlabel("Human Score")
        ax2.set_ylabel("LLM Score")
        ax2.set_xticks([1, 3, 5])
        ax2.set_yticks([1, 3, 5])
        ax2.set_xlim(0.7, 5.3)
        ax2.set_ylim(0.7, 5.3)
        # Stats annotation box
        ax2.text(1.01, 4.8, agreement_str, fontsize=13, va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#777", alpha=0.92))

    # --- Finalize Figure ---
    plt.tight_layout(rect=[0, 0.025, 1, 0.965])
    plt.savefig(PNG_FILE_PATH, dpi=320, bbox_inches='tight')
    plt.savefig(PDF_FILE_PATH, bbox_inches='tight')
    print(f"Saved visualization to:\n  {os.path.abspath(PNG_FILE_PATH)}\n  {os.path.abspath(PDF_FILE_PATH)}")


if __name__ == "__main__":
    main()
