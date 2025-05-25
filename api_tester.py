import io
import base64
import time
import requests
import json
from pathlib import Path

def load_test_audio():
    """Load the test audio file as base64."""
    with open("assets/exampleaudio.mp3", "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def save_audio(base64_audio: str, output_path: str):
    """Save base64 audio to file."""
    audio_bytes = base64.b64decode(base64_audio)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

def main():
    # Test text
    test_text = "Hello! This is a test of the Zonos API. How are you today? I hope everything is working well with the chunking and context-aware generation."
    
    # Load test audio as base64
    ref_audio_base64 = load_test_audio()
    
    # Create output directory
    output_dir = Path("api_test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("Testing Zonos API...")
    print(f"Text: {test_text}")
    
    # Prepare API request
    api_url = "http://localhost:4111/tts"
    payload = {
        "text": test_text,
        "ref_audio": ref_audio_base64,
        "language": "en-us"
    }
    
    # Measure generation time
    start_time = time.time()
    
    try:
        # Send request to API
        response = requests.post(api_url, json=payload, timeout=120)
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if 'audio' in result:
                # Save the audio
                output_path = output_dir / "api_test_output.wav"
                save_audio(result['audio'], str(output_path))
                
                print(f"✅ Success! Audio saved to {output_path}")
                print(f"⏱️  Generation time: {generation_time:.2f}s")
            else:
                print(f"❌ Error in response: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API is running on localhost:4111")
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 120 seconds")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 