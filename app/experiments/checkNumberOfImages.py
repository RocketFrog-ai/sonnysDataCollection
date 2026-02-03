import os

def count_png_files(directory_path: str) -> int:
    """
    Counts the number of .png files in a given directory and all its subdirectories.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        int: The total number of .png files found.
             Returns -1 and prints an error if the directory_path is invalid or not a directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory or does not exist.")
        return -1

    png_count = 0
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith('.png'):
                png_count += 1
    return png_count

if __name__ == "__main__":
    # Get directory path from user or use a default
    # target_dir = input("Enter the directory path to search: ")
    target_dir = "./collectedImages"  # Example: Search in the current directory

    # --- You can create a dummy directory structure for testing ---
    # import shutil
    # test_root = "temp_png_test_dir"
    # if os.path.exists(test_root):
    #     shutil.rmtree(test_root) # Clean up previous test
    # os.makedirs(os.path.join(test_root, "subdir1", "subsubdir"), exist_ok=True)
    # os.makedirs(os.path.join(test_root, "subdir2"), exist_ok=True)
    #
    # # Create some dummy files
    # open(os.path.join(test_root, "image1.png"), "w").close()
    # open(os.path.join(test_root, "document.txt"), "w").close()
    # open(os.path.join(test_root, "subdir1", "image2.PNG"), "w").close() # Test case-insensitivity
    # open(os.path.join(test_root, "subdir1", "subsubdir", "image3.png"), "w").close()
    # open(os.path.join(test_root, "subdir2", "another.jpg"), "w").close()
    # open(os.path.join(test_root, "subdir2", "final_image.PnG"), "w").close() # Test mixed case
    #
    # target_dir = test_root # Set target_dir to the test directory
    # --- End of dummy directory creation ---

    print(f"Searching for PNG files in: {os.path.abspath(target_dir)}")
    number_of_pngs = count_png_files(target_dir)

    if number_of_pngs != -1:
        print(f"Found {number_of_pngs} PNG file(s) in '{target_dir}' and its subdirectories.")

    # --- Clean up dummy directory if created ---
    # if target_dir == test_root and os.path.exists(test_root):
    #     shutil.rmtree(test_root)
    #     print(f"Cleaned up test directory: {test_root}")
    # --- End of cleanup ---