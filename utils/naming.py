from datetime import datetime

def build_filename(config, epoches, stage, extension=".pth", timestamped=False):
    base = f"{config['experiment']}_{config['model']}_{config['dataset']}_{config['aspect']}"

    # Fallbacks for missing config fields (e.g. in eval mode)
    lr_str = str(config.get('lr', 'NA')).replace('.', '') if config.get('lr') is not None else 'NA'
    bs_str = str(config.get('batch_size', 'NA'))
    n_ctx_str = str(config.get('n_ctx', 'NA'))

    if stage == "prompt":
        base += f"_prompt_nctx{n_ctx_str}_e{epoches}_lr{lr_str}_bs{bs_str}"
    elif stage == "image":
        base += f"_finetune_e{epoches}_lr{lr_str}_bs{bs_str}"

    if timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base += f"_{ts}"

    return base + extension
