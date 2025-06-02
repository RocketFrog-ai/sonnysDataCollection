import base64
from openai import OpenAI, AzureOpenAI
# from langchain_openai import AzureChatOpenAI
import os

client = OpenAI(api_key='5sPwzEUN6KlaevseHDUJ4CAt733wG7bJUuSTpssVV9GtB5Lyq7QKJQQJ99BDACHYHv6XJ3w3AAABACOGbWjz')

# client = OpenAI(api_key='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')
# client = AzureChatOpenAI(
#         model="gpt-4o",
#         api_version='2024-08-01-preview',
#         azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', 'https://talktodocs.openai.azure.com/'),
#         api_key=os.getenv('AZURE_OPENAI_KEY', 'B7UevKDPxBlJWO0BQrTvMvv1ZCcMiL1DyicvUg3k2170CLQ6LNPdJQQJ99BBACYeBjFXJ3w3AAABACOGcQhq'),
#     )

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "../notFoundData/collegeSelfServe.png"

# Getting the Base64 string
base64_image = encode_image(image_path)


response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "This is a satellite image of a car wash. Identify whether it has a tunnel or bay wash facility." },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

print(response.output_text)



