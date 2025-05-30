import os

# Exclude any of these paths or folder names
EXCLUDED_DIRS = {
    '.venv', '__pycache__', '.git',
    'datasets/11khands', 'datasets/HD'
}
EXCLUDED_NAMES = {'archived', 'archive'}

OUTPUT_FILE = "code_structure.txt"

def is_excluded(path, base_dir):
    rel_path = os.path.relpath(path, base_dir).replace("\\", "/")
    name = os.path.basename(path)
    if name in EXCLUDED_NAMES:
        return True
    for excl in EXCLUDED_DIRS:
        if rel_path.startswith(excl):
            return True
    return False

def walk_dir(base_dir, current_dir, depth=0):
    lines = []
    try:
        entries = sorted(os.listdir(current_dir))
    except Exception:
        return []

    for entry in entries:
        full_path = os.path.join(current_dir, entry)
        if is_excluded(full_path, base_dir):
            continue

        indent = "    " * depth
        if os.path.isdir(full_path):
            lines.append(f"{indent}{entry}/")
            lines.extend(walk_dir(base_dir, full_path, depth + 1))
        else:
            lines.append(f"{indent}{entry}")
    return lines

if __name__ == "__main__":
    base = os.getcwd()
    tree_lines = ["."]
    tree_lines += walk_dir(base, base)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))
    print(f"✅ Full directory + file structure saved to '{OUTPUT_FILE}'")
