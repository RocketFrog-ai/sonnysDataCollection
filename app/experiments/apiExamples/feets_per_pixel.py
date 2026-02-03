import math

# --- Constants ---
# This is the standard constant for Google's Web Mercator projection.
GOOGLE_MAPS_CONSTANT = 156543.03392

# The conversion factor from meters to feet.
METERS_TO_FEET = 3.28084

# The latitude of the geographic center of the continental US (near Lebanon, KS).
US_AVERAGE_LATITUDE = 39.8

def get_scale_for_location(latitude: float, longitude: float, zoom: int = 20) -> float:
    """
    Calculates the scale (feet per pixel) for a specific geographic location.

    This "score" tells you how much real-world distance in feet is covered
    by a single pixel in a Google Static Map satellite image at a given zoom level.

    Args:
        latitude (float): The latitude of the site (-90 to 90).
        longitude (float): The longitude of the site (-180 to 180).
                           Note: Longitude is not used in the scale calculation
                           but is included for function clarity, as you need both
                           coordinates to specify a location.
        zoom (int): The zoom level of the map (typically 0-21). Defaults to 20.

    Returns:
        float: The scale of the image in feet per pixel.
    """
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    # The Web Mercator projection's scale is dependent only on latitude.
    # We must convert latitude from degrees to radians for the math.cos() function.
    meters_per_pixel = GOOGLE_MAPS_CONSTANT * math.cos(math.radians(latitude)) / (2**zoom)

    # Convert the result from meters to feet to get the final score.
    feet_per_pixel = meters_per_pixel * METERS_TO_FEET

    return feet_per_pixel

def get_us_average_scale(zoom: int = 20) -> float:
    """
    Calculates a representative average scale (feet per pixel) for the continental US.

    This uses the latitude of the geographic center of the United States. It's a
    useful general-purpose score if you don't have the specific latitude of a site.

    Args:
        zoom (int): The zoom level of the map. Defaults to 20.

    Returns:
        float: The average scale for the US in feet per pixel.
    """
    # This is a convenience function that calls the main one with the US average latitude.
    # The longitude is arbitrary (0.0) as it's not used in the calculation.
    return get_scale_for_location(latitude=US_AVERAGE_LATITUDE, longitude=0.0, zoom=zoom)


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Calculating for a Specific Site ---")
    # A site in Miami, FL (southern US)
    site_lat = 41.83956
    site_lon = -87.65870
    zoom_level = 20

    specific_score = get_scale_for_location(site_lat, site_lon, zoom=zoom_level)
    print(f"Score: {specific_score:.4f} feet per pixel\n")
    
    # # Let's use this score
    # pixel_measurement = 300 # Imagine you measured a building to be 300px wide
    # estimated_length = pixel_measurement * specific_score
    # print(f"A {pixel_measurement}px object at this location is approx. {estimated_length:.2f} feet long.\n")
    
    # print("-" * 35)

    # print("\n--- Getting the General US Average Score ---")
    # average_us_score = get_us_average_scale(zoom=zoom_level)
    # print(f"The average score for the continental US at zoom {zoom_level} is:")
    # print(f"Score: {average_us_score:.4f} feet per pixel")
    # print(f"(This is calculated using a latitude of {US_AVERAGE_LATITUDE}° N)\n")
    
    # # Let's use the average score on the same 300px object
    # estimated_length_avg = pixel_measurement * average_us_score
    # print(f"Using the average score, the {pixel_measurement}px object would be estimated at {estimated_length_avg:.2f} feet.")
    # print("Notice the difference in estimation, showing why specific latitude is more accurate.")