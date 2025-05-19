
# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)

from google.genai import types
from google import genai


# client = genai.Client(api_key="AIzaSyAt59WZAmoN2FVj_FZM6wYvAdJa5Q3MFL0")
client = genai.Client(api_key="408dd71674a433cf696bb5109e8e2a183b211b02")


with open('notFoundData/collegeSelfServe.png', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
model='gemini-2.5-flash-preview-04-17',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/jpeg',
    ),
    'This is a satellite image of a car wash company. Decide whether it has tunnel wash facility or not.'
]
)

print(response.text)
