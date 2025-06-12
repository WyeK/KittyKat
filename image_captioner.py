import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def encode_image(image_path):
    """
    Encode an image file to base64.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_response(image_path, prompt, system_message, model="gpt-4.1"):
    """
    Get a response from OpenAI's API with an image.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): The prompt to send to the API
        system_message (str): The system message to set the assistant's behavior
        model (str): The model to use (defaults to GPT-4.1)
    
    Returns:
        str: The response from the API
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response from OpenAI: {e}")
        return None

def save_response_to_file(response, image_path):
    """
    Save the API response to a text file.
    
    Args:
        response (str): The response from the API
        image_path (str): Path to the original image file
    
    Returns:
        str: Path to the saved text file
    """
    # Create output directory if it doesn't exist
    output_dir = "captions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the same filename as the image but with .txt extension
    image_name = os.path.basename(image_path)
    output_filename = os.path.splitext(image_name)[0] + ".txt"
    
    # Save the response to file
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response)
    
    return output_path

if __name__ == "__main__":
    # Get all image files from the images directory
    image_dir = "images"
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    
    # Create images directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    
    # Read system message
    with open("ic_system_message.txt", "r", encoding="utf-8") as f:
        system_message = f.read()
    
    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, filename)
            print(f"\nProcessing image: {filename}")
            
            image_prompt = "This is an image from Birkenstock's Pinterest page. Describe what you see in this image"
            image_response = get_image_response(image_path, image_prompt, system_message)
            
            if image_response:
                print(f"Image Response: {image_response}")
                # Save the response to a file
                saved_file = save_response_to_file(image_response, image_path)
                print(f"Response saved to: {saved_file}")
            else:
                print(f"Failed to process image: {filename}")
