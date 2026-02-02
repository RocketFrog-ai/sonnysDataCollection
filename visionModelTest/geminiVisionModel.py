from google.genai import types
from google import genai


# client = genai.Client(api_key="AIzaSyAt59WZAmoN2FVj_FZM6wYvAdJa5Q3MFL0")
client = genai.Client(api_key="AIzaSyATs0pjterfjMVLOdzmC6mBQLCYw5oWhgw")
# client = genai.Client(vertexai=True, api_key="AIzaSyDq7-QEyUIezRe5xMf7LxsQBmnNOmRnfho")


with open('/home/arpit/dataCollection/collectedImages/Autobahn_Express__1225_Guyer_Ave__Richland__WA__99352/ChIJ1RPAbtN6mFQRrheeTWiaft0.png', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
model='gemini-3-flash-preview',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/jpeg',
    ),
    # 'This is a satellite image of a car wash, decide whether it has tunnel wash facility or not. A tunnel is a rectangular like structure with clear entry and exit points for a car drive-through. The response should always begin with a "Yes," or a "No," so that a "," can be used for spilitting.'
    '''
Classify a location as a “Competitor” if any of the following conditions are true:
Tunnel Wash Structure Identified


The image contains a long, narrow building that appears to function as a tunnel car wash.


The tunnel must have:


A clear entry and exit point, located at opposite narrow ends of the structure.


Possible wet patches or visible runoff on the pavement at the exit, indicating frequent vehicle washing.


Name Contains Competitive Keywords


If the car wash’s visible signage, label, or metadata includes any of the following words, it should be classified as a competitor:


"express"


"exterior"


"full serve"


"flex serve"


Vacuum Station Presence (Optional)


The image may show a vacuum station (typically small bays or rows with hoses/canisters), but this is not a requirement for classification.


If present, include it as a supportive detail.





Response Format:

Classification: Competitor / Not a Competitor

Justification:
- [Describe what structural or textual elements were identified that led to your classification.]
- [Mention if tunnel was detected, entry/exit presence, wet area, keyword in name, vacuum station, etc.]


Instructions:
Use all available visual cues (building layout, road markings, pavement conditions, signage) and any textual data or labels included with the image to make an informed judgment.

'''
]
)

print(response.text)
