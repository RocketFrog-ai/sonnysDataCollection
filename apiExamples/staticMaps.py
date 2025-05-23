import requests

# Replace with your Google Static Maps API key
API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

# Example coordinates from your JSON data
latitude = 33.858181
longitude = -84.67257769999999

# Google Static Maps API endpoint
base_url = "https://maps.googleapis.com/maps/api/staticmap"

# Parameters for the request
params = {
    "center": f"{latitude},{longitude}",
    "zoom": 20,               # Higher zoom = more detail (max ~21)
    "size": "640x640",        # Max allowed size without premium plan
    "scale": 2,
    "maptype": "hybrid",   # Use satellite imagery
    "key": API_KEY
}

# Make the GET request
response = requests.get(base_url, params=params)

# Save the image if the request was successful
if response.status_code == 200:
    with open("satellite_image.png", "wb") as f:
        f.write(response.content)
    print("Satellite image saved as 'satellite_image.png'")
else:
    print(f"Failed to retrieve image. Status code: {response.status_code}")
    print(response.text)