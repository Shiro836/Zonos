import argparse
import json
import os
import textwrap
import torch
import math
import logging
import warnings
warnings.filterwarnings(action='ignore', message=r".*eprecated.*") # category=FutureWarning, 
import torchaudio
import torch.nn.functional as F

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

logger = logging.getLogger(__name__)

def load_audio(file_paths, model):
    """Loads and preprocesses multiple audio files, left-padding them to align to the right."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # Convert single file path to list

    wavs = []
    max_length = 0  # Track max length to determine padding

    # Load and preprocess each file
    for file_path in file_paths:
        wav, sr = torchaudio.load(file_path)

        logger.debug(f"File: {file_path} | Shape: {wav.shape} | Sample Rate: {sr} | Channels: {wav.shape[0]}")

        if wav.shape[0] == 2:
            logger.warning(f"File: {file_path} | WARN: Prefix audio is stereo! Converting to mono")
            wav = wav.mean(0, keepdim=True)  # Convert to mono if needed
        
        if sr != 44_100:
            logger.warning(f"File: {file_path} | WARN: Prefix audio is {sr} Hz. Resampling to 44.1 kHz.")
            wav = torchaudio.functional.resample(wav, sr, 44_100)
        
        wavs.append(wav)

    max_length = max(math.ceil(w.shape[-1] / 512) * 512 for w in wavs)

    # Pad all waveforms to align them to the right
    padded_wavs = []
    for i, wav in enumerate(wavs):
        left_pad = max_length - wav.shape[-1]
        logger.debug(f"File: {file_paths[i]} | Left Padding: {left_pad} samples")  # Debugging info
        
        padded_wav = torch.nn.functional.pad(wav, (left_pad, 0), value=0)  # Left padding only
        padded_wavs.append(padded_wav)

    # Stack into a batch and encode
    batch_wav = torch.stack(padded_wavs).to(device, dtype=torch.float32)
    return model.autoencoder.encode(batch_wav)

def generate_audio(args, model, speaker_embedding, prefix_audio_codes, prefix_audio_text):
    """Generates speech for multiple texts in a batch."""

    batch_size = len(args.text)

    # Expand prefix audio codes for batch size
    prefix_audio_codes = prefix_audio_codes.repeat(batch_size // prefix_audio_codes.shape[0], 1, 1)

    # Expand prefix audio text for batch size
    prefix_audio_text = prefix_audio_text * (batch_size // len(prefix_audio_text))

    text = [(prefix_audio_text[i] + " " if prefix_audio_text[i] else "") + args.text[i] for i in range(batch_size)]

    torch.manual_seed(args.seed)
    logging.info(f"Seed: {args.seed}")

    # Create conditioning dictionaries for each text
    cond_dict = make_cond_dict(
        text=text,
        speaker=speaker_embedding,
        language=args.language,
        emotion=args.emotion,
        fmax=args.fmax,
        pitch_std=args.pitch_std,
        speaking_rate=args.speaking_rate,
        vqscore_8=args.vqscore_8,
        ctc_loss=args.ctc_loss,
        dnsmos_ovrl=args.dnsmos_ovrl,
        speaker_noised=args.speaker_noised,
        unconditional_keys=args.unconditional_keys,
    )

    prefix_conditioning = model.prepare_conditioning(cond_dict)

    # Generate codes
    codes = model.generate(
        prefix_conditioning,
        audio_prefix_codes=prefix_audio_codes,
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        batch_size=batch_size,
        disable_torch_compile=True, # When batching with these big tensors, torch.compile() makes things substantially slower
        sampling_params={
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "linear": args.linear,
            "conf": args.conf,
            "quad": args.quad,
            "repetition_penalty": args.repetition_penalty,
            "repetition_penalty_window": args.repetition_penalty_window,
            "temperature": args.temperature,
        },
        progress_bar=args.progress_bar,
    )

    for i, code in enumerate(codes):
        # Determine the padding lengths
        pad_i = len(str(len(args.text)-1))  # Number of digits in max_i
        output_file = f"{args.output.rstrip('.wav')}_{i:0{pad_i}d}.wav"

        # Ensure directory exists
        if len(os.path.dirname(output_file)) > 0:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save the decoded audio
        wavs = model.autoencoder.codes_to_wavs(code)
        if len(wavs) == 0:
            print(f"Generation of audio for {textwrap.shorten(text[i], width=75)} failed.")
            continue
            
        wav = wavs[0]
        sr = model.autoencoder.sampling_rate
        torchaudio.save(output_file, wav, sr)
        print(f"Generated audio saved to {output_file} | {textwrap.shorten(text[i], width=75)}")

def main():
    parser = argparse.ArgumentParser(description="Generate speech with Zonos CLI (Batch Mode).")
    parser.add_argument("--text", nargs="+", help="List of texts to generate speech for.")
    parser.add_argument("--language", default="en-us", help="Language code (e.g., en-us, de).")
    parser.add_argument("--reference_audio", default="assets/exampleaudio.mp3", help="Path to reference speaker audio.")
    parser.add_argument("--prefix_audio", "--audio_prefix", nargs="+", default=None, help="Path to prefix audio (default: 350ms silence).")
    parser.add_argument("--output", default="output.wav", help="Output wav file prefix.")
    parser.add_argument("--seed", type=int, default=423, help="Random seed for reproducibility.")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output.")
    parser.add_argument('--verbose_sampling', action='store_true', help="Print verbose sampling output.")
 
    # Conditioning parameters
    parser.add_argument("--emotion", nargs=8, type=float, default=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], help="Emotion vector (Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral).")
    parser.add_argument("--fmax", type=float, default=22050.0, help="Max frequency (0-24000).")
    parser.add_argument("--pitch_std", type=float, default=45.0, help="Pitch standard deviation (0-400).")
    parser.add_argument("--speaking_rate", type=float, default=15.0, help="Speaking rate (0-40).")
    parser.add_argument("--vqscore_8", nargs=8, type=float, default=[0.78] * 8, help="VQScore per 1/8th of audio.")
    parser.add_argument("--ctc_loss", type=float, default=0.0, help="CTC loss target.")
    parser.add_argument("--dnsmos_ovrl", type=float, default=4.0, help="DNSMOS overall score.")
    parser.add_argument("--speaker_noised", action='store_true', help="Apply speaker noise.")
    parser.add_argument("--unconditional_keys", nargs='*', default=["emotion", "vqscore_8", "dnsmos_ovrl"], help="Unconditional keys.")
        
    # Model generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=86 * 30, help="Max new tokens.")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG scale.")
    parser.add_argument("--top_p", type=float, default=0, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--min_p", type=float, default=0, help="Minimum probability threshold.")
    parser.add_argument("--linear", type=float, default=0.65, help="Linear scaling factor.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence parameter.")
    parser.add_argument("--quad", type=float, default=0.0, help="Quadratic factor.")
    parser.add_argument("--repetition_penalty", type=float, default=2.5, help="Repetition penalty.")
    parser.add_argument("--repetition_penalty_window", type=int, default=8, help="Repetition penalty window.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling.")
    parser.add_argument("--progress_bar", default=True, action="store_true", help="Show progress bar.")
    
    args = parser.parse_args()

    if args.text is None:
        raise ValueError("Please provide --text with texts to generate speech for.")

    if args.verbose_sampling:
        args.verbose = True
        logging.getLogger("zonos.sampling").setLevel(logging.DEBUG)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        logging.getLogger("phonemizer").setLevel(logging.DEBUG)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    from pytictoc import TicToc
    t = TicToc()
    t.tic()

    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    # model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device) # only GPU
    t.toc("Loading complete in", restart=True)
    
    print("Loading speaker reference audio...")
    wav, sr = torchaudio.load(args.reference_audio)
    speaker_embedding = model.make_speaker_embedding(wav, sr)
    t.toc("Speaker embedding complete in", restart=True)
    
    print("Loading prefix audio...")
    if args.prefix_audio:
        prefix_audio_codes = load_audio(args.prefix_audio, model)

        # Load associated transcripts from transcript.json
        if os.path.exists("transcripts.json"):
            with open("transcripts.json", "r") as f:
                transcript = json.load(f)

                prefix_audio_text = []
                for prefix_audio in args.prefix_audio:
                    key = os.path.splitext(os.path.basename(prefix_audio))[0]
                    if key not in transcript:
                        print(f"⚠️  Warning: Key '{key}' not found in transcript.json. Using empty string as prefix text.")
                    prefix_audio_text.append(transcript.get(key, ""))
        else:
            print("⚠️  Warning: transcripts.json not found. Using empty string as prefix text for all prefix audio.")
            prefix_audio_text = [""] * len(args.prefix_audio)
    else:
        silence_path = "assets/silence_100ms.wav"  # Ensure this file exists
        prefix_audio_codes = load_audio(silence_path, model)
        prefix_audio_text = [""]
    t.toc("Prefix audio complete in", restart=True)
    
    print("Generating speech...")
    generate_audio(args, model, speaker_embedding, prefix_audio_codes, prefix_audio_text)
    t.toc("Generating speech complete in", restart=True)
    
if __name__ == "__main__":
    main()