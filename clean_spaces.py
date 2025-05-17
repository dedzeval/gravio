import os
import glob

def clean_trailing_spaces(file_path):
    """Remove trailing spaces from each line in a file and save it back."""
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    #import pdb; pdb.set_trace()
    lines = [line[:-1] if line.endswith('\n') else line  for line in lines ]
    clr = [line for line in lines if line.rstrip() != line]
    cleaned_count =len(clr)
    # Remove trailing spaces
    cleaned_lines = [line.rstrip() + '\n' for line in lines]

    # If the file doesn't end with a newline, remove the last one
    if lines and not lines[-1].endswith('\n'):
        cleaned_lines[-1] = cleaned_lines[-1].rstrip()

    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    return len(lines), cleaned_count

def main():
    # Find all .js files in js/ directory
    js_files = glob.glob('js/*.js') + glob.glob('./*.py')

    if not js_files:
        print("No JavaScript files found in js/ directory")
        return

    # Clean each file
    total_files = 0
    total_lines = 0

    for file_path in js_files:
        try:
            lines, cleaned_count = clean_trailing_spaces(file_path)
            total_files += 1
            if not cleaned_count:
                continue
            total_lines += lines
            print(f"Cleaned {file_path} {cleaned_count} of{lines} lines")
        except Exception as e:
            print(f"Error cleaning {file_path}: {e}")

    print(f"\nTotal: {total_files} files processed, {total_lines} lines cleaned")

if __name__ == "__main__":
    main()