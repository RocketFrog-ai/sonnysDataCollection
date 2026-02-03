import re

reference_company_names = ['Whistle Express Car Wash',
 'Mister Car Wash',
 'Tidal Wave Auto Spa',
 'Quick Quack Express Car Wash',
 'Club Car Wash',
 'Tommy�s Express Car Wash',
 'Spotless Brands',
 'GO Car Wash',
 'Mammoth Holdings LLC',
 'WhiteWater Express Car Wash',
 'ModWash',
 'Express Wash Concepts',
 'Super Star Car Wash',
 'Autobell Car Wash',
 'LUV Car Wash',
 'Summit Wash Holdings',
 'True Blue Car Wash',
 'Splash Car Wash',
 'Caliber Car Wash',
 'BlueWave Express Car Wash',
 'El Car Wash',
 'Splash in ECO Car Wash',
 'Golden Nozzle Car Wash',
 'Champion Xpress Carwash',
 'Crew Carwash',
 "Sam's Xpress Car Wash",
 'Jax Kar Wash',
 'Terrible Herbst',
 "Big Dan's Car Wash",
 'Mr. Clean Car Wash',
 'Raceway Express Car Wash',
 'Sparkling Image Car Wash',
 "Wash N' Roll",
 "Rich's Car Wash",
 "Mike's Car Wash",
 'Surf Thru Express Car Wash',
 'Watershed Carwash',
 'WetGo Car Wash',
 'Delta Sonic',
 'The Wash Tub',
 'Gas N Wash',
 'Hurricane Express Wash',
 'Brown Bear Car Wash',
 'Waterway Carwash',
 'Hoffman Car Wash',
 'Quick N Clean',
 'ClearWater Express Wash',
 'Thoroughbred Express Auto Wash',
 'Trademark Car Wash',
 'Rocket Carwash',
 'Soapy Joe�s Car Wash',
 'Wash Masters Car Wash',
 'Washville Car Wash',
 'Fast 5 Xpress Car Wash',
 'Glide XPRESS Car Wash',
 'Prestige Car Wash',
 'Ultra Clean Express',
 'Flagstop Car Wash',
 'Mighty Wash',
 'Mr. Magic Car Wash',
 'ScrubaDub Auto Wash Centers',
 'Splash Car Wash & Oil Change',
 "Woodie's Wash Shack",
 'WOW Carwash',
 'Carnation Auto Spa',
 'Dirty Dog�s Car Wash',
 'Tagg-N-Go',
 'Drive & Shine Car Wash',
 'Flash Car Wash',
 'Sgt. Clean�s Car Wash',
 'TruShine Car Wash',
 'Francis & Sons Car Wash',
 "Fuller's Car Wash",
 'Shiny Shell Car Wash',
 'Washman Car Washes',
 'Bubble Down Car Wash',
 'Elephant Super Car Wash',
 'Kaady Car Wash',
 "Ronny's Car Wash",
 'Big Peach Car Wash',
 'Bliss Car Wash',
 'Buddy Bear Car Wash',
 "Fast Eddie's Car Wash & Oil Change",
 'Gate Express Car Wash',
 'Zax Auto Wash',
 "Haffner's",
 'Rainforest Carwash & Oil Change',
 'Shammy Shine Car Washes',
 'Car Spa',
 'Mach-1 Express Wash',
 'Ridi Stores',
 'Rocky Mountain Car Wash',
 'Breeze Thru Car Wash',
 'Carriage House Car Wash',
 'EDGE Express Car Wash',
 'Metro Express Car Wash',
 'Pelican Pointe Car Wash',
 'The Auto Spa',
 'Valet Auto Wash',
 'All American Car Wash',
 'Charlie�s Car Wash',
 'iShine Express Car Wash & Detail',
 'Jacksons',
 'Speedy Stop Car Wash',
 'Team Car Wash',
 'Wash-Rite Express Car Wash',
 'WashU Car Wash',
 'White Horse Auto Wash',
 'Fast Splash Car Wash',
 'Hang 10 Car Wash',
 'Raindrop Car Wash',
 'RocketFast Car Wash',
 'Super Clean Car Wash',
 'Time To Shine Car Wash',
 'WashX Car Wash',
 'Xtreme Auto Wash',
 '7 Flags Car Wash',
 'Cheetah Clean Auto Wash',
 "Ducky's Car Wash",
 'Groove Car Wash',
 'Mr Wash Car Wash',
 'Red Carpet Car Wash',
 'Scrub-a-Dub',
 'Shine Time Super Wash',
 'Tsunami Express',
 'Wash Factory Car Wash',
 'Xpress Car Wash',
 'Blue Penguin Car Wash',
 'BriteworX Car Washery',
 'Clean Sweep Car Wash',
 'Cloud 10 Smartwash',
 'Everclean Car Wash',
 'Scenic Suds Car Wash',
 'Scrubby�s Car Wash',
 'Skweeky Kleen Car Wash',
 'SpeedWash Car Wash',
 'Suds Deluxe Car Wash',
 'WashGuys Car Wash',
 'Auto Spa Car Wash (WA)',
 'Benny�s Car Wash',
 'Driven Car Wash',
 'Fabulous Freddy�s Car Wash',
 'Jet Splash Car Wash',
 'Radiant Express Car Wash',
 'Velocity Car Wash',
 'Auto Spa LA',
 'Clearwave Car Wash',
 'Ocean Blue Car Wash',
 'Snap Clean Car Wash',
 'Speedwash',
 'Sud Stop Car Wash',
 'Ace Auto Wash',
 'BlueBird Express Car Wash',
 'Dream Clean Car Wash',
 'H2Go Car Wash',
 'Pony Express Carwash',
 'Washtopia Car Wash',
 'American Pride Xpress Carwash',
 'Canton Car Wash',
 'Endless Clean Car Wash',
 'Mr. Sparkle Car Wash',
 'Mr. Splash Car Wash',
 "Pete's Express Car Wash",
 'RaceWash',
 'Screaming Eagle  Express Car Wash',
 'Shur-Kleen Car Wash',
 'Spin Car Wash',
 'Supershine Xpress Carwash',
 '5 Minute Express Car Wash',
 'Cadillac Express Car Wash',
 'Fast Freddy�s Car Wash',
 'FullSpeed Automotive',
 'Magic Brush Car Wash',
 'Soapy Noble Car Wash',
 'Waves Express Car Wash',
 'Four Seasons Car Wash']

def normalize_name(name):
    """
    Normalizes a company name through a simplified process:
    1. Converts to lowercase.
    2. Cleans internal "noise" characters (e.g., ®, ™, apostrophes), preserving spaces.
    3. Normalizes whitespace (multiple spaces/tabs to single space, trims).
    4. Removes a predefined list of common trailing phrases from the end.
    5. Finally, removes all remaining non-alphanumeric characters (i.e., spaces) for a compact key.
    """
    if not isinstance(name, str) or not name.strip():  # Handle None, non-string, empty or whitespace-only strings
        return ''

    name_lower = name.lower()

    # Clean internal "noise" characters (non-alphanumeric, non-space)
    # This removes symbols like ®, ™, ', ' etc., but keeps spaces.
    semi_cleaned = re.sub(r"[^a-z0-9\s]", '', name_lower)

    # Normalize whitespace (multiple spaces/tabs to single space, trim)
    semi_cleaned_normalized_space = ' '.join(semi_cleaned.split())

    # Remove known trailing suffixes.
    # Order by length (descending) for correct matching of overlapping/sub-phrases.
    trailing_phrases_to_remove = [
        "car wash",
        "carwash",
        "express car wash",
        "express carwash",    
        "xpress car wash",
        "xpress carwash",
        "auto wash"  
        "express wash"
        "xpress wash"
        "car washes"            
    ]
    trailing_phrases_to_remove.sort(key=len, reverse=True)  # Important: longest match first

    final_name = semi_cleaned_normalized_space
    for phrase in trailing_phrases_to_remove:
        if final_name.endswith(phrase):
            # Remove the phrase and strip any leading/trailing whitespace from the remainder
            final_name = final_name[:-len(phrase)].strip()
            # After removing a major suffix, assume the remainder is the core brand.
            # Stop further suffix stripping for this name.
            break
    
    # Final normalization - remove all remaining non-alphanumeric (i.e., spaces)
    # This collapses multiple word parts into a single string, e.g., "tidal wave" -> "tidalwave".
    normalized = re.sub(r'[^a-z0-9]', '', final_name)
    
    return normalized


def build_normalized_name_database(company_names_list):
    """
    Normalizes a list of company names and builds a lookup database.
    
    Args:
        company_names_list (list): A list of company names as strings.
        
    Returns:
        tuple: A tuple containing (set of normalized names, dict mapping normalized to original names).
    """
    normalized_set = set()
    normalized_to_original_map = {}

    for original_name in company_names_list:
        if not isinstance(original_name, str) or not original_name.strip():
            continue
            
        normalized = normalize_name(original_name)
        if normalized:
            normalized_set.add(normalized)
            if normalized not in normalized_to_original_map:
                normalized_to_original_map[normalized] = original_name

    return normalized_set, normalized_to_original_map


def match_competitors(competitor_names):
    """
    Compares a list of competitor names with reference company names.
    Uses name normalization to enable matching despite variations.

    Args:
        competitor_names (list): A list of competitor names (strings).
        reference_company_names (list): A list of reference company names (strings).

    Returns:
        tuple: A tuple containing count of found, list of found, list of not found.
    """
    if not isinstance(competitor_names, list) or not isinstance(reference_company_names, list):
        return 0, [], list(competitor_names) if competitor_names is not None else []
        
    normalized_reference_set, normalized_to_original_map = build_normalized_name_database(reference_company_names)

    found_competitors = []
    not_found_competitors = []

    for competitor_name in competitor_names:
        if not isinstance(competitor_name, str) or not competitor_name.strip():
            not_found_competitors.append(competitor_name)
            continue

        normalized_competitor = normalize_name(competitor_name)

        if not normalized_competitor:
            not_found_competitors.append(competitor_name)
            continue

        if normalized_competitor in normalized_reference_set:
            found_competitors.append(competitor_name)
        else:
            not_found_competitors.append(competitor_name)

    return len(found_competitors), found_competitors, not_found_competitors


