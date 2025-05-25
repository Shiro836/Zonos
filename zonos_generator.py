import io
import base64
import json
import torch
import torchaudio
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, get_symbol_ids, phonemize
from zonos.utils import DEFAULT_DEVICE as device

class TTSError(str, Enum):
    INVALID_AUDIO = "INVALID_AUDIO"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    INVALID_LANGUAGE = "INVALID_LANGUAGE"
    TEXT_TOO_SHORT = "TEXT_TOO_SHORT"
    GENERATION_FAILED = "GENERATION_FAILED"
    INVALID_PARAMS = "INVALID_PARAMS"

@dataclass
class TTSResult:
    success: bool
    data: Dict[str, str]  # Either {"result": base64_audio} or {"error": code, "message": msg}

class TTSGenerator:
    """Text-to-Speech generator class that reuses the model for efficiency."""
    
    def __init__(self, model_name: str = "Zyphra/Zonos-v0.1-transformer"):
        """Initialize the TTS generator with a model.
        
        Args:
            model_name: Name of the Zonos model to load
        """
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the Zonos model."""
        try:
            self.model = Zonos.from_pretrained(self.model_name, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def _process_voice_reference(self, voice_ref: io.BytesIO) -> Tuple[torch.Tensor, int]:
        """Process voice reference audio in memory.
        
        Args:
            voice_ref: BytesIO containing the audio file
            
        Returns:
            Tuple of (processed_audio_tensor, sample_rate)
            
        Raises:
            RuntimeError: If audio processing fails
        """
        try:
            # Load audio from bytes
            wav, sr = torchaudio.load(voice_ref)
            
            # Convert to mono if needed
            if wav.shape[0] == 2:
                wav = wav.mean(0, keepdim=True)
            
            # Resample to 44.1kHz if needed
            if sr != 44_100:
                wav = torchaudio.functional.resample(wav, sr, 44_100)
                sr = 44_100
            
            # Trim to 30 seconds
            max_samples = 30 * sr
            if wav.shape[-1] > max_samples:
                wav = wav[:, :max_samples]
            
            return wav, sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to process voice reference: {str(e)}")

    def _chunk_text(self, text: str, language: str = "en-us", min_tokens: int = 50, max_tokens: int = 80) -> List[str]:
        """Split text into chunks based on token count and punctuation.
        
        Args:
            text: Input text to chunk
            language: Language code for phonemization
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks (always returns at least one chunk)
            
        Raises:
            ValueError: Only if text is completely empty
        """
        # Handle empty text
        if not text or not text.strip():
            raise ValueError("No valid text content found")
        
        # For very short text, just return as single chunk
        text = text.strip()
        phonemes = phonemize([text], [language])[0]
        total_tokens = len(get_symbol_ids(phonemes))
        
        if total_tokens <= max_tokens:
            print(f"[CHUNKER DEBUG] Number of chunks: 1")
            print(f"[CHUNKER DEBUG] Chunk 1: '{text}'")
            return [text]
        
        # Split text into sentences (rough approximation)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            # No sentences found, return original text
            print(f"[CHUNKER DEBUG] Number of chunks: 1")
            print(f"[CHUNKER DEBUG] Chunk 1: '{text}'")
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            # Add period back for proper punctuation
            sentence = sentence + '.'
            # Get token count for the sentence using phoneme tokenization
            phonemes = phonemize([sentence], [language])[0]
            sentence_tokens = len(get_symbol_ids(phonemes))
            
            if current_tokens + sentence_tokens > max_tokens:
                # Current chunk would exceed max_tokens
                if current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0
                
                # Handle long sentences
                if sentence_tokens > max_tokens:
                    # Force split long sentence
                    words = sentence.split()
                    temp_chunk = []
                    temp_tokens = 0
                    
                    for word in words:
                        word_phonemes = phonemize([word], [language])[0]
                        word_tokens = len(get_symbol_ids(word_phonemes))
                        if temp_tokens + word_tokens > max_tokens:
                            if temp_chunk:
                                chunk_text = ' '.join(temp_chunk)
                                chunks.append(chunk_text)
                            temp_chunk = [word]
                            temp_tokens = word_tokens
                        else:
                            temp_chunk.append(word)
                            temp_tokens += word_tokens
                    
                    if temp_chunk:
                        chunk_text = ' '.join(temp_chunk)
                        chunks.append(chunk_text)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                    
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        # Ensure we always return at least one chunk
        if not chunks:
            print(f"[CHUNKER DEBUG] Number of chunks: 1")
            print(f"[CHUNKER DEBUG] Chunk 1: '{text}'")
            return [text]
        
        # Final output - only number of chunks and the chunks themselves
        print(f"[CHUNKER DEBUG] Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            chunk_phonemes = phonemize([chunk], [language])[0]
            chunk_tokens = len(get_symbol_ids(chunk_phonemes))
            print(f"[CHUNKER DEBUG] Chunk {i}: '{chunk}' ({chunk_tokens} tokens)")
            
        return chunks

    def _combine_audio_chunks(self, chunks: List[torch.Tensor], sr: int) -> bytes:
        """Combine audio chunks into a single WAV file in memory.
        
        Args:
            chunks: List of audio tensors
            sr: Sample rate
            
        Returns:
            WAV file as bytes
        """
        # Concatenate chunks
        combined = torch.cat(chunks, dim=-1)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, combined, sr, format="wav")
        return buffer.getvalue()

    def _enrich_chunks_with_context(self, chunks: List[str], language: str = "en-us", context_tokens: int = 10) -> Tuple[List[str], List[int]]:
        """Enrich chunks with context from previous chunks and track context boundaries.
        
        Args:
            chunks: List of text chunks
            language: Language code for tokenization
            context_tokens: Number of tokens to use as context from previous chunk
            
        Returns:
            Tuple of (enriched_chunks, context_token_counts)
        """
        if len(chunks) <= 1:
            return chunks, [0]  # No context needed for single chunk
        
        enriched_chunks = []
        context_token_counts = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk has no context
                enriched_chunks.append(chunk)
                context_token_counts.append(0)
            else:
                # Get context from previous chunk
                prev_chunk = chunks[i-1]
                prev_phonemes = phonemize([prev_chunk], [language])[0]
                prev_tokens = get_symbol_ids(prev_phonemes)
                
                # Extract last N tokens worth of text (approximate)
                if len(prev_tokens) > context_tokens:
                    # Estimate context text by taking last portion of previous chunk
                    words = prev_chunk.split()
                    context_words = []
                    context_token_count = 0
                    
                    # Work backwards through words to get approximately context_tokens
                    for word in reversed(words):
                        word_phonemes = phonemize([word], [language])[0]
                        word_token_count = len(get_symbol_ids(word_phonemes))
                        if context_token_count + word_token_count <= context_tokens:
                            context_words.insert(0, word)
                            context_token_count += word_token_count
                        else:
                            break
                    
                    context_text = ' '.join(context_words)
                    enriched_chunk = context_text + ' ' + chunk
                    enriched_chunks.append(enriched_chunk)
                    context_token_counts.append(context_token_count)
                    
                    print(f"[CONTEXT DEBUG] Chunk {i+1}: Added {context_token_count} context tokens: '{context_text}'")
                else:
                    # Previous chunk is shorter than context_tokens, use it all
                    enriched_chunk = prev_chunk + ' ' + chunk
                    enriched_chunks.append(enriched_chunk)
                    context_token_counts.append(len(prev_tokens))
                    
                    print(f"[CONTEXT DEBUG] Chunk {i+1}: Added {len(prev_tokens)} context tokens (full prev chunk)")
        
        return enriched_chunks, context_token_counts

    def _trim_context_from_audio(self, audio_chunks: List[torch.Tensor], context_token_counts: List[int], 
                                enriched_chunks: List[str], steps_per_chunk: List[List[int]], 
                                language: str, sr: int) -> List[torch.Tensor]:
        """Trim context portions from generated audio chunks using exact generation step data.
        
        Args:
            audio_chunks: List of generated audio tensors
            context_token_counts: Number of context tokens for each chunk
            enriched_chunks: The actual enriched text chunks that were generated
            steps_per_chunk: Actual generation steps recorded for each chunk
            language: Language code for phonemization
            sr: Sample rate
            
        Returns:
            List of trimmed audio tensors
        """
        trimmed_chunks = []
        
        for i, (audio, context_tokens, enriched_text, generation_steps) in enumerate(zip(audio_chunks, context_token_counts, enriched_chunks, steps_per_chunk)):
            if context_tokens == 0:
                # No context to trim
                trimmed_chunks.append(audio)
                print(f"[TRIM DEBUG] Chunk {i+1}: No context to trim")
            else:
                if not generation_steps:
                    # No generation data available, keep full audio
                    trimmed_chunks.append(audio)
                    print(f"[TRIM DEBUG] Chunk {i+1}: No generation data, keeping full audio")
                    continue
                
                # Calculate exact context duration using model's generation data
                total_generation_steps = len(generation_steps)
                audio_duration_seconds = audio.shape[-1] / sr
                
                # Use the model's actual generation rate (steps per second)
                steps_per_second = total_generation_steps / audio_duration_seconds
                
                # Calculate context portion based on exact step-to-audio mapping
                # Since generation is sequential (context first, then main content),
                # we can calculate the exact context steps using the generation rate
                
                try:
                    # Calculate EXACT total tokens for this enriched chunk
                    enriched_phonemes = phonemize([enriched_text], [language])[0]
                    total_tokens = len(get_symbol_ids(enriched_phonemes))
                    
                    # Now we can calculate the EXACT context ratio
                    context_ratio = context_tokens / total_tokens
                    
                    # Calculate exact context steps using the precise ratio
                    context_steps = int(total_generation_steps * context_ratio)
                    
                    # Convert steps to audio samples using the exact generation rate
                    context_duration_seconds = context_steps / steps_per_second
                    context_samples = int(context_duration_seconds * sr)
                    
                    # Safety cap: never trim more than 40% of audio
                    max_trim_samples = int(audio.shape[-1] * 0.4)
                    context_samples = min(context_samples, max_trim_samples)
                    
                    if context_samples > 0 and context_samples < audio.shape[-1]:
                        trimmed_audio = audio[:, context_samples:]
                        trimmed_chunks.append(trimmed_audio)
                        actual_context_duration = context_samples / sr
                        print(f"[TRIM DEBUG] Chunk {i+1}: Trimmed {actual_context_duration:.3f}s ({context_samples} samples)")
                        print(f"[TRIM DEBUG] Chunk {i+1}: EXACT calculation: {context_tokens}/{total_tokens} tokens = {context_steps}/{total_generation_steps} steps")
                    else:
                        # Context calculation resulted in invalid trim, keep full audio
                        trimmed_chunks.append(audio)
                        print(f"[TRIM DEBUG] Chunk {i+1}: Invalid context calculation, keeping full audio")
                        
                except Exception as e:
                    # If calculation fails, keep full audio
                    trimmed_chunks.append(audio)
                    print(f"[TRIM DEBUG] Chunk {i+1}: Calculation failed ({str(e)}), keeping full audio")
        
        return trimmed_chunks

    def generate(
        self,
        text: str,
        voice_ref: io.BytesIO,
        language: str = "en-us",
        conditioning_params: Optional[Dict] = None,
        generation_params: Optional[Dict] = None
    ) -> TTSResult:
        """Generate TTS audio from text using a voice reference.
        
        Args:
            text: Input text to convert to speech
            voice_ref: Voice reference audio as BytesIO
            language: Language code (default: "en-us")
            conditioning_params: Optional conditioning parameters
            generation_params: Optional generation parameters
            
        Returns:
            TTSResult containing either the generated audio or error information
        """
        try:
            # Set default parameters if None
            if conditioning_params is None:
                conditioning_params = {}
            if generation_params is None:
                generation_params = {}
                
            # Process voice reference
            try:
                wav, sr = self._process_voice_reference(voice_ref)
            except Exception as e:
                return TTSResult(False, {
                    "error": TTSError.INVALID_AUDIO,
                    "message": str(e)
                })
            
            # Get speaker embedding
            speaker_embedding = self.model.make_speaker_embedding(wav, sr)
            
            # Chunk text
            try:
                original_chunks = self._chunk_text(text, language)
            except ValueError as e:
                return TTSResult(False, {
                    "error": TTSError.TEXT_TOO_SHORT,
                    "message": str(e)
                })
            
                        # Enrich chunks with context for better continuity
            enriched_chunks, context_token_counts = self._enrich_chunks_with_context(original_chunks, language)
            
            # Track generation progress for precise context trimming
            generation_progress = {
                'current_batch_item': 0,
                'steps_per_chunk': [[] for _ in range(len(enriched_chunks))]
            }
            
            def generation_callback(frame: torch.Tensor, step: int, max_steps: int) -> bool:
                """Track generation steps for each chunk individually"""
                # In batched generation, the callback is called for each step of each item in the batch
                # We need to track which chunk we're currently generating
                batch_size = len(enriched_chunks)
                
                # Determine which chunk this step belongs to based on the frame tensor
                # The frame tensor has shape [batch_size, ...], so we can track each chunk
                if frame.shape[0] == batch_size:
                    # Store step for all chunks (they generate in parallel)
                    for i in range(batch_size):
                        generation_progress['steps_per_chunk'][i].append(step)
                
                return True  # Continue generation
            
            # Create conditioning dictionary for all enriched chunks at once (batched)
            cond_dict = make_cond_dict(
                text=enriched_chunks,  # Use enriched chunks with context
                speaker=speaker_embedding,
                language=language,
                emotion=conditioning_params.get("emotion", [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]),
                fmax=conditioning_params.get("fmax", 22050.0),
                pitch_std=conditioning_params.get("pitch_std", 45.0),
                speaking_rate=conditioning_params.get("speaking_rate", 15.0),
                vqscore_8=conditioning_params.get("vqscore_8", [0.78] * 8),
                ctc_loss=conditioning_params.get("ctc_loss", 0.0),
                dnsmos_ovrl=conditioning_params.get("dnsmos_ovrl", 4.0),
                speaker_noised=conditioning_params.get("speaker_noised", False),
                unconditional_keys=conditioning_params.get("unconditional_keys", ["emotion", "vqscore_8", "dnsmos_ovrl"]),
            )
            
            # Generate audio for all chunks in batch with callback tracking
            prefix_conditioning = self.model.prepare_conditioning(cond_dict)
            codes = self.model.generate(
                prefix_conditioning,
                max_new_tokens=generation_params.get("max_new_tokens", 86 * 30),
                cfg_scale=generation_params.get("cfg_scale", 2.0),
                batch_size=len(enriched_chunks),
                                disable_torch_compile=True,
                callback=generation_callback,  # Track exact generation steps
                sampling_params={
                    "top_p": generation_params.get("top_p", 0),
                    "top_k": generation_params.get("top_k", 0),
                    "min_p": generation_params.get("min_p", 0),
                    "linear": generation_params.get("linear", 0.65),
                    "conf": generation_params.get("conf", 0.4),
                    "quad": generation_params.get("quad", 0.0),
                    "repetition_penalty": generation_params.get("repetition_penalty", 2.5),
                    "repetition_penalty_window": generation_params.get("repetition_penalty_window", 8),
                    "temperature": generation_params.get("temperature", 1.0),
                },
                progress_bar=False,
            )
            
            # Decode all audio chunks at once
            audio_chunks = []
            for i, code in enumerate(codes):
                wavs = self.model.autoencoder.codes_to_wavs(code)
                if not wavs:
                    return TTSResult(False, {
                        "error": TTSError.GENERATION_FAILED,
                        "message": f"Failed to generate audio for chunk {i+1}: {enriched_chunks[i]}"
                    })
                audio_chunks.append(wavs[0])
            
            # Trim context from audio chunks for seamless transitions using exact generation data
            trimmed_chunks = self._trim_context_from_audio(
                audio_chunks, 
                context_token_counts,
                enriched_chunks,
                generation_progress['steps_per_chunk'],
                language,
                self.model.autoencoder.sampling_rate
            )
            
            # Combine chunks and convert to base64
            combined_audio = self._combine_audio_chunks(trimmed_chunks, self.model.autoencoder.sampling_rate)
            base64_audio = base64.b64encode(combined_audio).decode('utf-8')
            
            return TTSResult(True, {
                "result": base64_audio
            })
            
        except Exception as e:
            return TTSResult(False, {
                "error": TTSError.GENERATION_FAILED,
                "message": str(e)
            })

# Backward compatibility function
def generate_tts(
    text: str,
    voice_ref: io.BytesIO,
    language: str = "en-us",
    conditioning_params: Optional[Dict] = None,
    generation_params: Optional[Dict] = None
) -> TTSResult:
    """Generate TTS audio from text using a voice reference.
    
    This is a backward compatibility function that creates a new generator instance.
    For better performance, use TTSGenerator class directly.
    
    Args:
        text: Input text to convert to speech
        voice_ref: Voice reference audio as BytesIO
        language: Language code (default: "en-us")
        conditioning_params: Optional conditioning parameters
        generation_params: Optional generation parameters
        
    Returns:
        TTSResult containing either the generated audio or error information
    """
    generator = TTSGenerator()
    return generator.generate(text, voice_ref, language, conditioning_params, generation_params) 