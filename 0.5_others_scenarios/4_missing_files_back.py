import os

def get_files_with_keyword(directory, keywords):
    """
    Recursively retrieves a list of files in a directory whose paths contain any of the specified keywords.
    
    Args:
        directory (str): The root directory to search.
        keywords (list): A list of keywords to filter files.

    Returns:
        list: A list of relative file paths containing the keywords.
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), directory)
            if any(keyword in file_path for keyword in keywords):
                matching_files.append(file_path)
    return matching_files

def compare_and_find_missing(reference_files, target_files):
    """
    Compares two lists of file paths and determines which files from the reference list are missing in the target list.
    
    Args:
        reference_files (list): List of reference file paths.
        target_files (list): List of target file paths.

    Returns:
        list: A list of missing file paths.
    """
    return list(set(reference_files) - set(target_files))

def main():
    # Define the directories and keywords
    reference_directory = "input/frames_all_without_back_without_blur/"  # Replace with your reference directory path
    target_directory = "input/frames_all_with_back_without_blur/3_ddbs-s_with_back_without_blur/"  # Replace with your target directory path
    keywords = ["evk4", "davis346"]
    
    # Get matching files from both directories
    reference_files = get_files_with_keyword(reference_directory, keywords)
    target_files = get_files_with_keyword(target_directory, keywords)
    
    # Compare and find missing files
    missing_files = compare_and_find_missing(reference_files, target_files)
    
    # Save missing files to a .txt file
    output_file = "missing_files.txt"
    with open(output_file, "w") as f:
        for file in missing_files:
            f.write(file + "\n")
    
    print(f"Missing files written to {output_file}")

if __name__ == "__main__":
    main()
