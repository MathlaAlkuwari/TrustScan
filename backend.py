import os
import io
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from anthropic import Anthropic

app = Flask(__name__)
# Enable CORS for cross-origin requests from the Lovable frontend
CORS(app)

# Initialize the Anthropic Client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Path to your authentic reference receipt image (Absolute Path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "real_reference.jpg")


def encode_image_to_base64(image):
    """
    Converts a PIL Image object to a base64 encoded string.
    """
    buffered = io.BytesIO()
    # Convert image to RGB if necessary to ensure JPEG compatibility
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route('/analyze', methods=['POST'])
def analyze_receipt():
    # Check if a file was included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['file']

    try:
        # Load the user-uploaded image
        user_image = Image.open(io.BytesIO(uploaded_file.read()))
        user_image_b64 = encode_image_to_base64(user_image)

        # Load the reference image stored on the server
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            return jsonify({"error": f"Reference image not found at {REFERENCE_IMAGE_PATH}"}), 500

        ref_image = Image.open(REFERENCE_IMAGE_PATH)
        ref_image_b64 = encode_image_to_base64(ref_image)

        # System prompt instructions for Claude
        prompt_instructions = """
You are a senior forensic expert in financial document verification, specifically analyzing Bahrain "Fawri+" bank transfer receipts.

I have provided two images:
1. First Image: An authentic reference receipt showing valid fonts, alignments, logo proportions, element spacing, and official layouts.
2. Second Image: A newly uploaded receipt that needs to be inspected for potential tampering or forgery.

IMPORTANT NOTICE:
Dynamic variable data (such as account numbers, phone numbers, recipient names, transfer amounts, dates, and reference numbers) will naturally vary between receipts. Do NOT classify a receipt as fraudulent simply because these values differ.

Your focus must strictly be on technical visual artifacts, including:
- Font family consistency, weight, and stroke thickness.
- Text alignment, line spacing, and boundary padding.
- Pixelation artifacts, image compression inconsistencies, or photo editing marks around numerical fields.
- Logo distortions or altered background patterns.

You MUST respond strictly with a valid JSON object matching the following structure:
{
  "is_authentic": true or false,
  "risk_score_percentage": number between 0 and 100 (where 0 is completely authentic and 100 is definitely fraudulent),
  "reasons": [
    "Detailed technical reason 1",
    "Detailed technical reason 2"
  ]
}

Return ONLY the raw JSON object without markdown formatting, code blocks, or extra text.
"""

        # Call the Claude 3.5 Sonnet Vision API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": ref_image_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": user_image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_instructions
                        }
                    ],
                }
            ]
        )

        # Extract textual response
        raw_content = response.content[0].text.strip()

        # Parse JSON response to validate format
        parsed_json = json.loads(raw_content)

        return jsonify(parsed_json), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON response from Claude API", "raw_response": raw_content}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
