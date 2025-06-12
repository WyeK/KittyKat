import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

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
    
def parse_response(response_text):
    """
    Parse the response text as JSON.
    
    Args:
        response_text (str): The response text from the API
    
    Returns:
        dict: Parsed JSON response
    """
    try:
        # Try to parse the response as JSON
        parsed_response = json.loads(response_text)
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

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
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response from OpenAI: {e}")
        return None
    
def save_json_response(json_data, image_path):
    """
    Save the JSON response to a file with the same name as the image.
    
    Args:
        json_data (dict): The JSON data to save
        image_path (str): Path to the original image file
    
    Returns:
        str: Path to the saved JSON file
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = "tags"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the same filename as the image but with .json extension
        image_name = os.path.basename(image_path)
        output_filename = os.path.splitext(image_name)[0] + ".json"
        
        # Save the JSON to file
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        
        return output_path
    except Exception as e:
        print(f"Error saving JSON response: {e}")
        return None

def aggregate_tags(tags_list):
    """
    Aggregate tags and their weights, combining duplicates and averaging weights.
    
    Args:
        tags_list (list): List of tag dictionaries
    
    Returns:
        list: Aggregated list of unique tags with averaged weights
    """
    tag_dict = {}
    for tag in tags_list:
        tag_name = tag['tag']
        weight = tag['weight']
        if tag_name in tag_dict:
            # Average the weights for duplicate tags
            tag_dict[tag_name] = (tag_dict[tag_name] + weight) / 2
        else:
            tag_dict[tag_name] = weight
    
    # Convert back to list format
    return [{"tag": k, "weight": v} for k, v in tag_dict.items()]

if __name__ == "__main__":
    # Read system message
    with open("it_system_message.txt", "r", encoding="utf-8") as f:
        system_message = f.read()
    
    # Initialize aggregated tags
    aggregated_tags = {
        "lighting": [],
        "setting": [],
        "style": [],
        "grain": [],
        "mood": [],
        "framing": [],
        "character": [],
        "emotions": []
    }
    
    # Get all image files from the images directory
    image_dir = "images"
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    
    # Create images directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    
    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, filename)
            print(f"\nProcessing image: {filename}")
            
            image_prompt = "This is an image from Birkenstock's Pinterest page. Tag this image"
            image_response = get_image_response(image_path, image_prompt, system_message)
            
            if image_response:
                parsed_response = parse_response(image_response)
                if parsed_response:
                    # Save individual image tags
                    saved_file = save_json_response(parsed_response, image_path)
                    if saved_file:
                        print(f"JSON response saved to: {saved_file}")
                    
                    # Aggregate tags
                    for category in aggregated_tags.keys():
                        if category in parsed_response:
                            aggregated_tags[category].extend(parsed_response[category])
            else:
                print(f"Failed to process image: {filename}")
    
    # Aggregate and sort tags for each category
    final_tags = {}
    for category, tags in aggregated_tags.items():
        if tags:  # Only process categories that have tags
            aggregated = aggregate_tags(tags)
            # Sort by weight in descending order
            sorted_tags = sorted(aggregated, key=lambda x: x['weight'], reverse=True)
            final_tags[category] = sorted_tags
    
    # Save aggregated tags
    if final_tags:
        output_path = os.path.join("tags", "aggregated_tags.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_tags, f, indent=2)
        print(f"\nAggregated tags saved to: {output_path}")
        print("\nAggregated Tags Summary:")
        print(json.dumps(final_tags, indent=2))
