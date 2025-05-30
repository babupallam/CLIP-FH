import os
import re
import pandas as pd

def extract_stage3_metrics(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    metrics = {}

    # Format 1  PromptSG style
    pattern1_splits = re.findall(
        r'Split (\d+)/10\s+Evaluation Metrics:\s+'
        r'Rank-1\s*:\s*([\d.]+)%\s+'
        r'Rank-5\s*:\s*([\d.]+)%\s+'
        r'Rank-10\s*:\s*([\d.]+)%\s+'
        r'mAP\s*:\s*([\d.]+)%', content)

    pattern1_final = re.search(
        r'Final Averaged Results Across All Splits:\s*'
        r'Rank-1 Accuracy\s*:\s*([\d.]+)%\s*'
        r'Rank-5 Accuracy\s*:\s*([\d.]+)%\s*'
        r'Rank-10 Accuracy\s*:\s*([\d.]+)%\s*'
        r'Mean AP\s*:\s*([\d.]+)%', content)

    # Format 2  ReID-style logs
    pattern2_splits = re.findall(
        r'\[Split (\d+)\] Rank-1: ([\d.]+)%, Rank-5: ([\d.]+)%, Rank-10: ([\d.]+)%, mAP: ([\d.]+)%', content)

    pattern2_final = re.search(
        r'FINAL EVAL SUMMARY.*?'
        r'Avg Rank-1\s*:\s*([\d.]+)%\s*'
        r'Avg Rank-5\s*:\s*([\d.]+)%\s*'
        r'Avg Rank-10\s*:\s*([\d.]+)%\s*'
        r'Mean AP\s*:\s*([\d.]+)%', content, re.DOTALL)

    # Use matching pattern
    if pattern1_splits:
        for split in pattern1_splits:
            i = int(split[0])
            metrics[f'split{i}_R1'] = float(split[1])
            metrics[f'split{i}_R5'] = float(split[2])
            metrics[f'split{i}_R10'] = float(split[3])
            metrics[f'split{i}_mAP'] = float(split[4])
    elif pattern2_splits:
        for split in pattern2_splits:
            i = int(split[0])
            metrics[f'split{i}_R1'] = float(split[1])
            metrics[f'split{i}_R5'] = float(split[2])
            metrics[f'split{i}_R10'] = float(split[3])
            metrics[f'split{i}_mAP'] = float(split[4])

    if pattern1_final:
        metrics['final_R1'] = float(pattern1_final.group(1))
        metrics['final_R5'] = float(pattern1_final.group(2))
        metrics['final_R10'] = float(pattern1_final.group(3))
        metrics['final_mAP'] = float(pattern1_final.group(4))
    elif pattern2_final:
        metrics['final_R1'] = float(pattern2_final.group(1))
        metrics['final_R5'] = float(pattern2_final.group(2))
        metrics['final_R10'] = float(pattern2_final.group(3))
        metrics['final_mAP'] = float(pattern2_final.group(4))

    return metrics

def collect_stage3_results(base_dir='eval_logs', num_versions=11):
    vitb16_results = {}
    rn50_results = {}

    for i in range(1, num_versions + 1):  # Fix: include stage3-v11
        version = f"stage3-v{i}"
        folder = os.path.join(base_dir, version)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if fname.endswith(".log"):
                metrics = extract_stage3_metrics(fpath)
                if 'vitb16' in fname.lower():
                    vitb16_results[version] = metrics
                elif 'rn50' in fname.lower():
                    rn50_results[version] = metrics

    vitb16_df = pd.DataFrame.from_dict(vitb16_results, orient='index')
    rn50_df = pd.DataFrame.from_dict(rn50_results, orient='index')
    vitb16_df.index.name = "version"
    rn50_df.index.name = "version"

    return vitb16_df.sort_index(), rn50_df.sort_index()

# Run and save
vitb16_df, rn50_df = collect_stage3_results('eval_logs', num_versions=11)

os.makedirs("result_logs", exist_ok=True)
vitb16_df.to_csv("result_logs/stage3_vitb16_eval_table.csv")
rn50_df.to_csv("result_logs/stage3_rn50_eval_table.csv")

print("Stage 3 tables (unified format) saved to result_logs/")
