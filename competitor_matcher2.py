import sys
import re
import csv

# Global cache for normalized CSV data
_csv_data_cache = {}

def normalize_name(name):
    """
    Normalizes a company name through a multi-step process:
    1. Converts to lowercase.
    2. Splits at common structural separators like '|', '(', '-' to get a core candidate.
    3. Cleans internal "noise" characters (e.g., ®, ™, apostrophes) from this candidate, preserving spaces.
    4. Normalizes whitespace (multiple spaces/tabs to single space, trims).
    5. Removes a predefined list of common trailing phrases from the end.
    6. Finally, removes all remaining non-alphanumeric characters (i.e., spaces) for a compact key.
    """
    if not isinstance(name, str) or not name.strip(): # Handle None, non-string, empty or whitespace-only strings
        return ''

    name_lower = name.lower()

    # Step 1: Extract core name candidate by splitting at structural separators
    # These often separate the main brand from generic descriptors or location info.
    # \uff5c is the full-width vertical bar, sometimes seen in data.
    core_name_candidate = re.split(r'\s*[\(|\uff5c|-]\s*', name_lower, 1)[0].strip()

    # Step 2: Clean internal "noise" characters (non-alphanumeric, non-space)
    # This removes symbols like ®, ™, ', ’ etc., but keeps spaces.
    # Essential for examples like "Tommy's Express®" vs "Tommy’s Express"
    semi_cleaned_core = re.sub(r"[^a-z0-9\s]", '', core_name_candidate)

    # Step 3: Normalize whitespace (multiple spaces/tabs to single space, trim)
    # Ensures consistent spacing before suffix matching. E.g., "word  word" -> "word word"
    semi_cleaned_core_normalized_space = ' '.join(semi_cleaned_core.split())

    # Step 4: Remove known trailing suffixes.
    # The list should contain phrases as they would appear after Steps 2 & 3.
    # (e.g., "&" would have been removed, "and" would be preserved).
    # Order by length (descending) for correct matching of overlapping/sub-phrases.
    trailing_phrases_to_remove = [
        "express car wash",
        "express carwash",            # Variant of "express car wash"
        "full service car wash",
        # Consider "full service carwash" if it's a common pattern.
        "car wash and detail center", # For inputs like "... and Detail Center"
        "car wash detail center",     # For inputs like "... & Detail Center" (after & removed by Step 2)
        "car wash and lube center",   # For inputs like "... and Lube Center"
        "car wash lube center",       # For inputs like "... & Lube Center" (after & removed by Step 2)
        "car wash",
        "carwash",
        # "auto spa", "express" (alone) are intentionally omitted as they can be
        # integral parts of a brand name, not just generic suffixes.
    ]
    trailing_phrases_to_remove.sort(key=len, reverse=True) # Important: longest match first

    final_core_name = semi_cleaned_core_normalized_space
    for phrase in trailing_phrases_to_remove:
        if final_core_name.endswith(phrase):
            # Remove the phrase and strip any leading/trailing whitespace from the remainder
            final_core_name = final_core_name[:-len(phrase)].strip()
            # After removing a major suffix, assume the remainder is the core brand.
            # Stop further suffix stripping for this name.
            break
    
    # Step 5: Final normalization - remove all remaining non-alphanumeric (i.e., spaces)
    # This collapses multiple word parts into a single string, e.g., "tidal wave" -> "tidalwave".
    normalized = re.sub(r'[^a-z0-9]', '', final_core_name)
    
    return normalized


def _load_and_normalize_csv_data(csv_filepath):
    """
    Loads, normalizes, and caches car wash company names from a CSV file.
    Uses the `csv` module for robust parsing.
    Returns a tuple: (set of normalized names, dict mapping normalized to original CSV names).
    Returns (None, None) on critical file errors that prevent processing.
    Returns (set(), {}) if the file is parsable but no valid data is found (e.g., no 'Company Name' column).
    """
    if csv_filepath in _csv_data_cache:
        cached_data = _csv_data_cache[csv_filepath]
        return cached_data.get('normalized_set', set()), cached_data.get('normalized_to_original_map', {})

    original_csv_names = []
    
    try:
        with open(csv_filepath, mode='r', encoding='cp1252', newline='') as f:
            reader = csv.reader(f)
            try:
                header_row = next(reader)
            except StopIteration:
                print(f"Warning: CSV file '{csv_filepath}' is empty or has no header.")
                _csv_data_cache[csv_filepath] = {'normalized_set': set(), 'normalized_to_original_map': {}}
                return set(), {}

            header = [h.strip().lower() for h in header_row]

            try:
                company_name_index = header.index('company name')
            except ValueError:
                print(f"Error: 'Company Name' column not found in the CSV file '{csv_filepath}'. Header: {header_row}")
                _csv_data_cache[csv_filepath] = {'normalized_set': set(), 'normalized_to_original_map': {}}
                return set(), {}

            for row_values in reader:
                if len(row_values) > company_name_index:
                    company_name_raw = row_values[company_name_index].strip()
                    if company_name_raw:
                        original_csv_names.append(company_name_raw)

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_filepath}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV '{csv_filepath}': {e}")
        return None, None

    normalized_set = set()
    normalized_to_original_map = {}

    for original_name in original_csv_names:
        normalized = normalize_name(original_name)
        if normalized:
            normalized_set.add(normalized)
            if normalized not in normalized_to_original_map:
                 normalized_to_original_map[normalized] = original_name

    _csv_data_cache[csv_filepath] = {
        'normalized_set': normalized_set,
        'normalized_to_original_map': normalized_to_original_map
    }
    return normalized_set, normalized_to_original_map


def match_competitors_with_csv(competitor_names, csv_filepath):
    """
    Compares a list of competitor names with car wash company names from a CSV file.
    Uses improved normalization and caches CSV data.

    Args:
        competitor_names (list or None): A list of competitor names (strings).
        csv_filepath (str): The path to the Car Wash Advisory Companies CSV file.

    Returns:
        tuple: A tuple containing count of found, list of found, list of not found.
    """
    normalized_car_wash_companies_set, _ = _load_and_normalize_csv_data(csv_filepath)

    if normalized_car_wash_companies_set is None:
        print(f"Could not load data from CSV: {csv_filepath}. Marking all competitors as not found.")
        return 0, [], list(competitor_names) if competitor_names is not None else []

    found_competitors_input_names = []
    not_found_competitors_input_names = []
    input_names_iterable = competitor_names if competitor_names is not None else []

    for competitor_input_name in input_names_iterable:
        if not isinstance(competitor_input_name, str) or not competitor_input_name.strip():
            not_found_competitors_input_names.append(competitor_input_name)
            continue

        normalized_competitor = normalize_name(competitor_input_name)

        if not normalized_competitor:
            not_found_competitors_input_names.append(competitor_input_name)
            continue

        if normalized_competitor in normalized_car_wash_companies_set:
            found_competitors_input_names.append(competitor_input_name)
        else:
            not_found_competitors_input_names.append(competitor_input_name)

    return len(found_competitors_input_names), found_competitors_input_names, not_found_competitors_input_names

if __name__ == "__main__":
    print("This script is intended to be imported and used by countCompetitors.py")
    print("Running internal tests for match_competitors_with_csv...")

    # Create a dummy CSV for testing
    dummy_csv_filepath = 'dummy_car_wash_companies.csv'
    with open(dummy_csv_filepath, 'w', newline='', encoding='cp1252') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'Company Name', 'Other Column'])
        writer.writerow(['1', 'Tidal Wave Auto Spa', 'Info1'])          # norm: tidalwaveautospa
        writer.writerow(['2', 'Mister Car Wash', 'Info2'])              # norm: mister
        writer.writerow(['3', 'Zips', 'Info3'])                         # norm: zips
        writer.writerow(['4', 'Super Express Wash', 'Info4'])           # norm: superexpresswash
        writer.writerow(['5', 'Go Carwash', 'Info5'])                   # norm: go
        writer.writerow(['6', 'Clean Getaway Car Wash & Detail Center', 'Info6']) # norm: cleangetaway
        writer.writerow(['7', 'Alpha Beta Express Car Wash', 'Info7'])  # norm: alphabeta
        writer.writerow(['8', '', 'No name here'])                      # norm: ''
        writer.writerow(['9', '  Whitespace   Car   Wash  ', 'Whitespace test']) # norm: whitespace
        writer.writerow(['10', 'Tommy’s Express Car Wash', 'InfoTommy'])# norm: tommys
        writer.writerow(['11', "Another Brand Express CARWASH", 'InfoBrand']) # norm: anotherbrand


    print(f"\n--- Test Case: Basic Match (Tidal Wave, Mister, Zips) ---")
    # Expected: Tidal Wave Auto Spa | Car Wash -> tidalwaveautospa
    #           Mister Car Wash -> mister
    #           Zips Car Wash -> zips
    competitors1 = ["Tidal Wave Auto Spa | Car Wash", "Mister Car Wash", "Non Existent Wash", "Zips Car Wash"]
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors1, dummy_csv_filepath)
    print(f"Found Count: {found_count}") # Expected: 3
    print(f"Found Competitors: {found_list}") # Expected: ['Tidal Wave Auto Spa | Car Wash', 'Mister Car Wash', 'Zips Car Wash']
    print(f"Not Found Competitors: {not_found_list}") # Expected: ['Non Existent Wash']

    print(f"\n--- Test Case: Normalization Variants (Super Express, Go, Clean Getaway, Alpha Beta) ---")
    # Expected:
    # "Super Express Wash (South Location)" -> superexpresswash
    # "GoCarwash" -> go
    # "Clean Getaway Car Wash and Detail Center" -> cleangetaway
    # " Tidal Wave Auto SPA " -> tidalwaveautospa
    # "Alpha Beta Express CARWASH" -> alphabeta (with "express carwash" in suffix list)
    competitors2 = [
        "Super Express Wash (South Location)",
        "GoCarwash",
        "Clean Getaway Car Wash and Detail Center",
        " Tidal Wave Auto SPA ",
        "Alpha Beta Express CARWASH"
    ]
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors2, dummy_csv_filepath)
    print(f"Found Count: {found_count}") # Expected: 5
    print(f"Found Competitors: {found_list}")
    print(f"Not Found Competitors: {not_found_list}") # Expected: []

    print(f"\n--- Test Case: Tommy's Express Match (Symbol and Apostrophe Variants) ---")
    # Expected: Both normalize to "tommys"
    # CSV has "Tommy’s Express Car Wash"
    competitors_tommy = ["Tommy's Express® Car Wash", "Tommy’s Express Car Wash"]
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors_tommy, dummy_csv_filepath)
    print(f"Found Count: {found_count}") # Expected: 2
    print(f"Found Competitors: {found_list}")
    print(f"Not Found Competitors: {not_found_list}") # Expected: []

    print(f"\n--- Test Case: 'express carwash' (no space) suffix ---")
    # CSV has "Another Brand Express CARWASH" -> anotherbrand
    # Competitor: "Another Brand Express Carwash" -> anotherbrand
    competitors_express_carwash = ["Another Brand Express Carwash"]
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors_express_carwash, dummy_csv_filepath)
    print(f"Found Count: {found_count}") # Expected: 1
    print(f"Found Competitors: {found_list}")
    print(f"Not Found Competitors: {not_found_list}")

    print(f"\n--- Test Case: Cache test (call again) ---")
    print("(This call should use cached CSV data)")
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors1, dummy_csv_filepath)
    print(f"Found Count (cached): {found_count}")
    print(f"Found Competitors (cached): {found_list}")
    print(f"Not Found Competitors (cached): {not_found_list}")

    # ... (other test cases for bad input, file not found, etc., remain the same) ...

    print(f"\n--- Test Case: Edge Cases for Input ---")
    competitors3 = [None, "", "  ", "Actual Competitor", 123]
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors3, dummy_csv_filepath)
    print(f"Found Count: {found_count}")
    print(f"Found Competitors: {found_list}")
    print(f"Not Found Competitors: {not_found_list}")

    print(f"\n--- Test Case: Non-existent CSV ---")
    found_count, found_list, not_found_list = match_competitors_with_csv(competitors1, "non_existent_file.csv")
    print(f"Found Count (bad CSV): {found_count}")
    print(f"Found Competitors (bad CSV): {found_list}")
    print(f"Not Found Competitors (bad CSV): {not_found_list}")

    # Clean up dummy file
    import os
    try:
        os.remove(dummy_csv_filepath)
        # os.remove(bad_header_csv) # If you re-add these tests
        # os.remove(empty_csv_path)
        # os.remove(header_only_csv)
    except OSError as e:
        print(f"Error removing test files: {e}")
    print("\nInternal tests finished.")