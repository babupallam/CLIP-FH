from datetime import datetime

def build_filename(config, epoches, stage, extension=".pth", timestamped=False):
    base = f"{config['experiment']}_{config['model']}_{config['dataset']}_{config['aspect']}"
    if stage == "prompt":
        base += f"_prompt_nctx{config['n_ctx']}_e{epoches}_lr{str(config['lr']).replace('.', '')}_bs{config['batch_size']}"
    elif stage == "image":
        base += f"_finetune_e{epoches}_lr{str(config['lr']).replace('.', '')}_bs{config['batch_size']}"
    if timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base += f"_{ts}"
    return base + extension


