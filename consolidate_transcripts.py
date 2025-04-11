import os
import argparse
from pathlib import Path
import difflib

def get_files_dict(folder_path):
    """Create a dictionary of files in the given folder."""
    files_dict = {}
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            files_dict[file] = file_path
    return files_dict

def read_file_content(file_path):
    """Read and return the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ""

def create_comparison_file(parent_dir):
    """
    Compare transcription files from two folders and create consolidated transcriptions.
    
    Args:
        parent_dir: Path to the parent directory containing the two transcription folders
    """
    # Identify the two transcription folders
    subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    
    if len(subfolders) < 2:
        print(f"Error: Found only {len(subfolders)} folders in {parent_dir}. Need at least 2 folders.")
        return
    
    folder1 = subfolders[0]
    folder2 = subfolders[1]
    
    # Get files from both folders
    files1 = get_files_dict(folder1)
    files2 = get_files_dict(folder2)
    
    # Create a set of all unique files from both folders
    all_files = set(files1.keys()).union(set(files2.keys()))
    
    if not all_files:
        print("No files found in the folders.")
        return
    
    # Determine model names from folder names
    model1_name = os.path.basename(os.path.normpath(folder1))
    model2_name = os.path.basename(os.path.normpath(folder2))
    
    # Create output directory for consolidated transcriptions
    output_dir = os.path.join(parent_dir, "consolidated_transcriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics tracking
    stats = {
        "total_files": len(all_files),
        "identical_content": 0,
        "only_model1_has_content": 0,
        "only_model2_has_content": 0,
        "both_empty": 0,
        "different_content": 0
    }
    
    # Keep track of files in different categories
    empty_files = []
    identical_files = []
    only_model1_files = []
    only_model2_files = []
    different_files = []
    
    # Process each file
    for file in sorted(all_files):
        content1 = read_file_content(files1.get(file, "")) if file in files1 else ""
        content2 = read_file_content(files2.get(file, "")) if file in files2 else ""
        
        # Apply the specified logic and track statistics
        if content1 and content2:
            if content1 == content2:
                # Both have identical content
                stats["identical_content"] += 1
                identical_files.append(file)
                final_content = content1
            else:
                # Both have different content, use Generated1
                stats["different_content"] += 1
                different_files.append(file)
                final_content = content1
        elif content1:
            # Only content1 has content
            stats["only_model1_has_content"] += 1
            only_model1_files.append(file)
            final_content = content1
        elif content2:
            # Only content2 has content
            stats["only_model2_has_content"] += 1
            only_model2_files.append(file)
            final_content = content2
        else:
            # Both are empty
            stats["both_empty"] += 1
            empty_files.append(file)
            final_content = ""
        
        # Write the final content to the output directory
        output_file_path = os.path.join(output_dir, file)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
    
    # Create a comparison file (TXT format only)
    output_file = os.path.join(parent_dir, f"comparison_{model1_name}_vs_{model2_name}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Add statistics at the top
        f.write(f"TRANSCRIPTION COMPARISON STATISTICS\n")
        f.write(f"=================================\n\n")
        f.write(f"Total files analyzed: {stats['total_files']}\n")
        f.write(f"Files with identical content: {stats['identical_content']} ({round(stats['identical_content'] / stats['total_files'] * 100, 1) if stats['total_files'] > 0 else 0}%)\n")
        f.write(f"Files with content only in {model1_name}: {stats['only_model1_has_content']} ({round(stats['only_model1_has_content'] / stats['total_files'] * 100, 1) if stats['total_files'] > 0 else 0}%)\n")
        f.write(f"Files with content only in {model2_name}: {stats['only_model2_has_content']} ({round(stats['only_model2_has_content'] / stats['total_files'] * 100, 1) if stats['total_files'] > 0 else 0}%)\n")
        f.write(f"Files with differences between sources: {stats['different_content']} ({round(stats['different_content'] / stats['total_files'] * 100, 1) if stats['total_files'] > 0 else 0}%)\n")
        f.write(f"Files with no content in either source: {stats['both_empty']} ({round(stats['both_empty'] / stats['total_files'] * 100, 1) if stats['total_files'] > 0 else 0}%)\n\n")
        
        f.write(f"Consolidated transcriptions created in: {output_dir}\n\n")
        f.write(f"{'=' * 80}\n\n")
        
        # Add file-by-file comparisons
        for file in sorted(all_files):
            content1 = read_file_content(files1.get(file, "")) if file in files1 else ""
            content2 = read_file_content(files2.get(file, "")) if file in files2 else ""
            
            f.write(f"File: {file}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"{model1_name} Transcription:\n{content1}\n\n")
            f.write(f"{model2_name} Transcription:\n{content2}\n\n")
            
            # Add selected content info
            if content1 and content2:
                if content1 == content2:
                    f.write(f"Status: IDENTICAL CONTENT\n")
                else:
                    f.write(f"Status: DIFFERENT CONTENT - Selected: {model1_name} (as specified)\n")
            elif content1:
                f.write(f"Status: CONTENT ONLY IN {model1_name} - Selected: {model1_name}\n")
            elif content2:
                f.write(f"Status: CONTENT ONLY IN {model2_name} - Selected: {model2_name}\n")
            else:
                f.write("Status: NO CONTENT IN EITHER SOURCE\n")
            
            # Add diff for files with differences
            if content1 and content2 and content1 != content2:
                diff = difflib.unified_diff(
                    content1.splitlines(), 
                    content2.splitlines(), 
                    lineterm='',
                    fromfile=model1_name,
                    tofile=model2_name
                )
                f.write("\nDifferences:\n")
                diff_text = '\n'.join(diff)
                f.write(diff_text)
            
            f.write(f"\n\n{'=' * 80}\n\n")
    
    # Create statistics file with detailed lists
    stats_file = os.path.join(parent_dir, "transcription_stats.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"TRANSCRIPTION COMPARISON STATISTICS\n")
        f.write(f"=================================\n\n")
        f.write(f"Total files analyzed: {stats['total_files']}\n\n")
        
        f.write(f"Files with identical content: {stats['identical_content']}\n")
        if identical_files:
            f.write("Files with identical content:\n")
            for file in sorted(identical_files):
                f.write(f"  - {file}\n")
            f.write("\n")
        
        f.write(f"Files with content only in {model1_name}: {stats['only_model1_has_content']}\n")
        if only_model1_files:
            f.write(f"Files with content only in {model1_name}:\n")
            for file in sorted(only_model1_files):
                f.write(f"  - {file}\n")
            f.write("\n")
        
        f.write(f"Files with content only in {model2_name}: {stats['only_model2_has_content']}\n")
        if only_model2_files:
            f.write(f"Files with content only in {model2_name}:\n")
            for file in sorted(only_model2_files):
                f.write(f"  - {file}\n")
            f.write("\n")
        
        f.write(f"Files with differences between sources: {stats['different_content']}\n")
        if different_files:
            f.write("Files with differences between sources:\n")
            for file in sorted(different_files):
                f.write(f"  - {file}\n")
            f.write("\n")
        
        f.write(f"Files with no content in either source: {stats['both_empty']}\n")
        if empty_files:
            f.write("Files with no content in either source:\n")
            for file in sorted(empty_files):
                f.write(f"  - {file}\n")
    
    # Print statistics to console
    print(f"\nTranscription Comparison Statistics:")
    print(f"Total files analyzed: {stats['total_files']}")
    print(f"Files with identical content: {stats['identical_content']}")
    print(f"Files with content only in {model1_name}: {stats['only_model1_has_content']}")
    print(f"Files with content only in {model2_name}: {stats['only_model2_has_content']}")
    print(f"Files with differences between sources: {stats['different_content']}")
    print(f"Files with no content in either source: {stats['both_empty']}")
    
    print(f"\nConsolidated transcriptions created in: {output_dir}")
    print(f"Comparison file created: {output_file}")
    print(f"Statistics file created: {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare and consolidate transcription files from two folders.')
    parser.add_argument('parent_dir', help='Path to the parent directory containing the two transcription folders')
    
    args = parser.parse_args()
    
    create_comparison_file(args.parent_dir)