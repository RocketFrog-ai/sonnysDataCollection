import requests
import os

# Replace with your Yelp API key
API_KEY = "kJ4dN5pciIj0FfUhVjT3KQAaIc6gzmtjKZ7cn23TqxyxPPHlJCbhnluB-ZsEqLYQYIJxNI4gaIzNE8n2iQi9lgGPVGFShfgPVkWqkD52AeHn9_0N-X6uMg6ZCoIwaHYx"
latitude = "32.8352883"  # Example coordinates
longitude = "-116.7654836"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

def save_image(url, filename):
    """Download and save an image from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")

# Step 1: Search for car wash businesses near coordinates
search_url = "https://api.yelp.com/v3/businesses/search"
search_params = {
    "latitude": latitude,
    "longitude": longitude,
    "term": "car wash",
    "limit": 1
}

response = requests.get(search_url, params=search_params, headers=headers)

if response.status_code == 200:
    businesses = response.json().get("businesses", [])
    if not businesses:
        print("No car wash businesses found.")
        exit()

    business = businesses[0]
    business_name = business["name"].replace(" ", "_")  # Sanitize name for filenames

    # Create a directory for the business's images
    os.makedirs(business_name, exist_ok=True)

    # Step 2: Extract photo URLs from the search response
    photos = business.get("photos", [])

    if not photos:
        print(f"No photos found for {business_name}.")
    else:
        print(f"Found {len(photos)} photos for {business_name}:")
        for idx, photo_url in enumerate(photos):
            filename = os.path.join(business_name, f"image_{idx+1}.jpg")
            save_image(photo_url, filename)

else:
    print(f"Error searching for businesses: {response.status_code} - {response.text}")