import os


def print_directory_structure(startpath):
    # First print the directory structure
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = "│   " * level
        print(f"{indent}├── {os.path.basename(root)}/")
        for f in files:
            print(f"{indent}│   ├── {f}")

    print("\n=== File Contents ===\n")

    # Then print the contents of each file
    for root, dirs, files in os.walk(startpath):
        for f in files:
            filepath = os.path.join(root, f)
            print(f"\n=== {filepath} ===\n")
            try:
                with open(filepath, "r") as file:
                    print(file.read())
            except Exception as e:
                print(f"Could not read file: {e}")


# Usage
print_directory_structure("./experiment")
