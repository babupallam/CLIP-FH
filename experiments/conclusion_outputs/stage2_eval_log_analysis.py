import os
import re
import pandas as pd

def extract_all_metrics(file_path):
    """Extracts all split-wise and final metrics from a .log file."""
    with open(file_path, 'r') as f:
        content = f.read()

    split_pattern = re.compile(
        r'Split (\d+)/10\s*'
        r'Rank-1\s*:\s*([\d.]+)%\s*'
        r'Rank-5\s*:\s*([\d.]+)%\s*'
        r'Rank-10\s*:\s*([\d.]+)%\s*'
        r'mAP\s*:\s*([\d.]+)%'
    )

    final_pattern = re.search(
        r'Rank-1 Accuracy\s*:\s*([\d.]+)%\s*\n'
        r'Rank-5 Accuracy\s*:\s*([\d.]+)%\s*\n'
        r'Rank-10 Accuracy\s*:\s*([\d.]+)%\s*\n'
        r'Mean AP\s*:\s*([\d.]+)%', content)

    metrics = {}
    # Extract split-wise
    for match in split_pattern.finditer(content):
        split = int(match.group(1))
        metrics[f'split{split}_R1'] = float(match.group(2))
        metrics[f'split{split}_R5'] = float(match.group(3))
        metrics[f'split{split}_R10'] = float(match.group(4))
        metrics[f'split{split}_mAP'] = float(match.group(5))

    # Extract final
    if final_pattern:
        metrics['final_R1'] = float(final_pattern.group(1))
        metrics['final_R5'] = float(final_pattern.group(2))
        metrics['final_R10'] = float(final_pattern.group(3))
        metrics['final_mAP'] = float(final_pattern.group(4))

    return metrics

def collect_results_by_model(base_dir='eval_logs'):
    vitb16_results = {}
    rn50_results = {}

    for i in range(1, 11):
        version = f"stage2-v{i}"
        folder = os.path.join(base_dir, version)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if fname.endswith(".log"):
                metrics = extract_all_metrics(fpath)
                if 'vitb16' in fname.lower():
                    vitb16_results[version] = metrics
                elif 'rn50' in fname.lower():
                    rn50_results[version] = metrics

    vitb16_df = pd.DataFrame.from_dict(vitb16_results, orient='index')
    rn50_df = pd.DataFrame.from_dict(rn50_results, orient='index')
    vitb16_df.index.name = "version"
    rn50_df.index.name = "version"

    return vitb16_df.sort_index(), rn50_df.sort_index()

# Save results
vitb16_df, rn50_df = collect_results_by_model('eval_logs')

# Ensure result_logs exists
os.makedirs("result_logs", exist_ok=True)

vitb16_df.to_csv("result_logs/stage2_vitb16_eval_table.csv")
rn50_df.to_csv("result_logs/stage2_rn50_eval_table.csv")

print("Saved tables to result_logs/")
