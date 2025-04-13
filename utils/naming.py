from datetime import datetime

def build_filename(config, epoches, stage, extension=".pth", timestamped=False):
    base = f"{config['experiment']}_{config['model']}_{config['dataset']}_{config['aspect']}"

    # Optional values with defaults
    lr_str = str(config.get('lr', 'NA')).replace('.', '') if config.get('lr') is not None else 'NA'
    bs_str = str(config.get('batch_size', 'NA'))
    n_ctx_str = str(config.get('n_ctx', 'NA'))
    variant = config.get('variant', 'unknown').lower()

    # Build filename based on variant
    if variant in ["stage1_frozen_text", "prompt"]:
        base += f"_prompt_nctx{n_ctx_str}_e{epoches}_lr{lr_str}_bs{bs_str}"
    elif variant in ["clipreid", "stage2", "joint", "finetuned"]:
        base += f"_finetune_e{epoches}_lr{lr_str}_bs{bs_str}"
    elif variant in ["promptsg", "stage3"]:
        base += f"_promptsg_e{epoches}_lr{lr_str}_bs{bs_str}"
    elif variant in ["baseline"]:
        base += f"_baseline_e{epoches}_bs{bs_str}"
    else:
        base += f"_{variant}_e{epoches}_bs{bs_str}"  # generic fallback

    if timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base += f"_{ts}"

    return base + extension
