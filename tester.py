import io
import base64
import time
import torchaudio
from pathlib import Path

from zonos_generator import TTSGenerator

def load_test_audio():
    """Load the test audio file as raw bytes."""
    with open("assets/exampleaudio.mp3", "rb") as f:
        return io.BytesIO(f.read())

def save_audio(base64_audio: str, output_path: str):
    """Save base64 audio to file."""
    audio_bytes = base64.b64decode(base64_audio)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

def get_audio_duration(base64_audio: str) -> float:
    """Get duration of audio from base64 encoded WAV data."""
    try:
        audio_bytes = base64.b64decode(base64_audio)
        # Create a temporary BytesIO object to load the audio
        audio_buffer = io.BytesIO(audio_bytes)
        wav, sr = torchaudio.load(audio_buffer)
        duration = wav.shape[-1] / sr
        return duration
    except Exception as e:
        print(f"Warning: Could not calculate audio duration: {e}")
        return 0.0

def main():
    # Test cases with different punctuation and length scenarios
    test_texts = [
        # Normal text with proper punctuation
        "Hello! This is a test. How are you today? I hope everything is going well.",
        
        # Very long sentence
        "This is an extremely long sentence that should be split into multiple chunks because it contains a lot of words and goes on and on without any punctuation to break it up which is not ideal for text to speech generation.",
        
        # Multiple short sentences
        "Hi. Hello. Hey. How are you? I'm good. Thanks.",
        
        # Mixed punctuation
        "What's up! This is cool... But wait, what about this? And this! And finally, this.",
        
        # Text with numbers and special characters
        "The price is $99.99! That's 50% off. Call now at 1-800-555-0123. Don't wait!!!",

        # Long text with multiple sentences
        "AI-Driven API Suggestion Tool: An application that uses AI to suggest optimal API architectures and design patterns based on specified project requirements and goals, focusing on performance, scalability, and ease of integration.Automated Database Optimization Service: A service that analyzes database schemas and queries, providing automated suggestions and modifications to improve performance and scalability for both SQL and NoSQL databases.",
    ]
    
    # Load test audio
    voice_ref = load_test_audio()
    
    # Create output directory
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create TTS generator (model is loaded once and reused)
    print("Loading TTS generator...")
    generator = TTSGenerator()
    print("Generator loaded successfully!")
    
    # Process each test case
    for i, text in enumerate(test_texts, 1):
        print(f"\nProcessing test case {i}:")
        print(f"Text: {text}")
        
        # Reset BytesIO position before each test
        voice_ref.seek(0)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate TTS using the reusable generator
        result = generator.generate(
            text=text,
            voice_ref=voice_ref,
            language="en-us"
        )
        
        generation_time = time.time() - start_time
        
        if result.success:
            # Save the audio
            output_path = output_dir / f"test_case_{i}.wav"
            save_audio(result.data["result"], str(output_path))
            
            # Calculate RTF (Real-Time Factor)
            audio_duration = get_audio_duration(result.data["result"])
            rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
            
            print(f"Success! Audio saved to {output_path}")
            print(f"Generation time: {generation_time:.2f}s")
            print(f"Audio duration: {audio_duration:.2f}s")
            print(f"RTF (Real-Time Factor): {rtf:.2f}x")
            if rtf < 1.0:
                print("✓ Faster than real-time!")
            else:
                print("⚠ Slower than real-time")
        else:
            print(f"Error: {result.data['error']}")
            print(f"Message: {result.data['message']}")
            print(f"Generation time: {generation_time:.2f}s")

if __name__ == "__main__":
    main() 