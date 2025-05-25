import logging
import json
import time
import io
import base64
import soundfile
from threading import Lock
from flask import Flask, jsonify, request
import torch

logging.getLogger().setLevel(logging.ERROR)

app = Flask("zonos_tts")

mutex = Lock()

from zonos_generator import TTSGenerator

# Initialize the generator once at startup
generator = None

def generate(text, ref_audio, language="en-us", conditioning_params=None, generation_params=None):
    """Generate TTS using Zonos with voice reference"""
    global generator
    if generator is None:
        generator = TTSGenerator()
    
    # Convert ref_audio bytes to BytesIO
    voice_ref = io.BytesIO(ref_audio)
    
    # Generate using our zonos generator
    result = generator.generate(
        text=text,
        voice_ref=voice_ref,
        language=language,
        conditioning_params=conditioning_params,
        generation_params=generation_params
    )
    
    if result.success:
        # Decode base64 audio back to bytes for soundfile processing
        audio_bytes = base64.b64decode(result.data["result"])
        
        # Load the WAV data and return as numpy array
        audio_data, sr = soundfile.read(io.BytesIO(audio_bytes))
        return audio_data, sr
    else:
        raise Exception(f"{result.data['error']}: {result.data['message']}")

@app.route('/tts', methods=['POST'])
def tts():
    start = time.time()
    text = request.json['text']
    ref_audio_base64 = request.json['ref_audio']
    
    # Optional parameters with defaults
    language = request.json.get('language', 'en-us')
    conditioning_params = request.json.get('conditioning_params', {})
    generation_params = request.json.get('generation_params', {})

    print("Got text from client: ", text)

    with mutex:
        try:
            audio_data, sr = generate(
                text, 
                base64.b64decode(ref_audio_base64), 
                language, 
                conditioning_params, 
                generation_params
            )

            # Convert audio back to WAV bytes
            file_object = io.BytesIO()
            soundfile.write(file_object, audio_data, sr, format="wav")
            file_object.seek(0)
            file_string = file_object.read()
            audio = base64.b64encode(file_string).decode('utf-8')
            
            print("Done in " + str(time.time() - start) + " seconds")

            return jsonify({'audio': audio})
        except Exception as e:
            print(f"Error in generate: {e}")
            return jsonify({'error': f"Error in generate: {e}"})
        finally:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

import os

def main():
    print("Warming up Zonos...")
    ref_file = "assets/exampleaudio.mp3"  # Use the existing reference file
    
    if os.path.exists(ref_file):
        with open(ref_file, 'rb') as file:
            ref = file.read()
        
        # Warmup with a long text to test chunking
        warmup_text = "Hello! This is a warmup test for the Zonos TTS system. We're testing multiple sentences to ensure chunking works properly. The system should handle this text efficiently and produce high-quality speech output."
        
        try:
            audio_data, sr = generate(warmup_text, ref)
            
            # Save warmup output
            output_file = "warmup_zonos_output.wav"
            soundfile.write(output_file, audio_data, sr, format="wav")
            print(f"Warmup complete! Output saved to {output_file}")
        except Exception as e:
            print(f"Warmup failed: {e}")
    else:
        print(f"Warning: Reference file {ref_file} not found. Skipping warmup.")

if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=4111, debug=False) 