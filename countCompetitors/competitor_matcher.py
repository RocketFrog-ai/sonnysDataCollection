import sys
import re

def normalize_name(name):
    """Normalizes a company name by removing spaces, symbols, and converting to lowercase."""
    if isinstance(name, str):
        name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
    return name

def match_competitors_with_csv(competitor_names, csv_filepath):
    """
    Compares a list of competitor names with car wash company names from a CSV file.

    Args:
        competitor_names (list): A list of competitor names.
        csv_filepath (str): The path to the Car Wash Advisory Companies CSV file.

    Returns:
        tuple: A tuple containing the count of found competitors, a list of found competitors, and a list of not found competitors.
    """
    car_wash_companies = []
    try:
        with open(csv_filepath, mode='r', encoding='cp1252') as f:
            # Read and process header
            first_line = f.readline()
            header = [h.strip() for h in first_line.strip().split(',')]

            try:
                company_name_index = header.index('Company Name')
            except ValueError:
                print("Error: 'Company Name' column not found in the CSV file.")
                return 0, [], competitor_names # Return 0 found, empty found list, and all competitors as not found

            # Process data rows
            for line in f:
                row = [cell.strip() for cell in line.strip().split(',')]
                if row and len(row) > company_name_index: # Ensure row is not empty and has enough columns
                    company_name = row[company_name_index]
                    if normalize_name(company_name): # Add check for empty name after normalization
                        car_wash_companies.append(company_name)

    except FileNotFoundError:
        print(f"Error: {csv_filepath} not found.")
        return 0, [], competitor_names # Return 0 found, empty found list, and all competitors as not found
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return 0, [], competitor_names # Return 0 found, empty found list, and all competitors as not found

    # Normalize the car wash company names
    normalized_car_wash_companies = [normalize_name(name) for name in car_wash_companies]

    # Compare competitor names with car wash company names
    found_count = 0
    found_competitors = []
    not_found_competitors = []

    for competitor in competitor_names:
        normalized_competitor = normalize_name(competitor)
        if normalized_competitor in normalized_car_wash_companies:
            found_count += 1
            found_competitors.append(competitor)
        else:
            not_found_competitors.append(competitor)

    return found_count, found_competitors, not_found_competitors

if __name__ == "__main__":
    # This part is for testing the function independently if needed
    # In the final solution, this script will be imported.
    print("This script is intended to be imported and used by countCompetitors.py")
    print("It can be tested here by providing a dummy list of competitor names and the CSV path.")
    # Example usage for testing:
    # dummy_competitors = ["Whistle Express Car Wash", "Non Existent Car Wash", "Mister Car Wash"]
    # csv_file = 'Car_Wash_Advisory_Companies.csv'
    # found_count, found_list, not_found_list = match_competitors_with_csv(dummy_competitors, csv_file)
    # print(f"\nFound Count: {found_count}")
    # print(f"Found Competitors: {found_list}")
    # print(f"Not Found Competitors: {not_found_list}")
