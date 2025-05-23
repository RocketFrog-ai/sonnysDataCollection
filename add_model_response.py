import pandas as pd
import os

def add_model_response_column(csv_file, collected_images_dir):
    """
    Adds a 'model response' column to the CSV file with content from corresponding text files.

    Args:
        csv_file (str): Path to the accuracyCheck.csv file.
        collected_images_dir (str): Path to the directory containing collected images subfolders.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return

    # Add the new column if it doesn't exist
    if 'model response' not in df.columns:
        df['model response'] = None

    last_folder_name = None

    for index, row in df.iterrows():
        folder_name = row['folderName']
        file_name = row['fileName']

        if pd.notna(folder_name):
            last_folder_name = folder_name

        if pd.notna(file_name) and last_folder_name:
            text_file_name = None
            if file_name.endswith('.png'):
                text_file_name = f"2.5-flash-{file_name.replace('.png', '.txt')}"
            elif file_name == 'matching_results.txt':
                text_file_name = 'matching_results.txt'

            if text_file_name:
                # Construct the full path to the text file
                text_file_path = os.path.join(collected_images_dir, last_folder_name, text_file_name)

                if os.path.exists(text_file_path):
                    try:
                        with open(text_file_path, 'r') as f:
                            model_response_content = f.read()
                        df.at[index, 'model response'] = model_response_content
                    except Exception as e:
                        print(f"Error reading file {text_file_path}: {e}")
                else:
                    print(f"Warning: Text file not found for {file_name} at {text_file_path}")

    # Function to extract 'Yes' or 'No'
    def extract_model_identification(row):
        if row['fileName'] == 'matching_results.txt':
            return None  # Leave blank for matching_results.txt rows
        model_response = row['model response']
        if pd.notna(model_response) and ',' in model_response:
            return model_response.split(',')[0].strip()
        return None # Handle cases where there's no response or no comma

    # Apply the function to create the new column
    df['model Identification'] = df.apply(extract_model_identification, axis=1)

    # Save the updated DataFrame back to the CSV
    try:
        df.to_csv(csv_file, index=False)
        print(f"Successfully updated {csv_file}")
    except Exception as e:
        print(f"Error writing to {csv_file}: {e}")

if __name__ == "__main__":
    csv_file_path = 'accuracyCheck.csv'
    collected_images_directory = 'collectedImages'
    add_model_response_column(csv_file_path, collected_images_directory)
