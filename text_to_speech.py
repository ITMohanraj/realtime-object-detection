# text_to_speech.py
import tempfile
import time
import os
import io
from gtts import gTTS

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import TTS_LANG, TTS_SLOW

class TextToSpeech:
    def __init__(self):
        """Initialize TTS engine, handling headless environments gracefully"""
        self.audio_available = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self.audio_available = True
            except Exception as e:
                print(f"Failed to initialize Pygame mixer (running in headless mode): {e}")
        else:
            print("Pygame is not installed. Desktop audio playback is disabled.")
    
    def speak(self, text, lang=TTS_LANG, slow=TTS_SLOW):
        """
        Convert text to speech and play it locally
        
        Args:
            text: Text to convert to speech
            lang: Language code
            slow: Whether to speak slowly
        
        Returns:
            True if successful, False otherwise
        """
        if not self.audio_available:
            print(f"Local audio playback not available. Speaking: '{text}'")
            return False
            
        try:
            if not text or text.strip() == "":
                return False
            
            # Generate TTS
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)
            
            # Play audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            os.remove(temp_file)
            
            return True
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return False
            
    def get_speech_bytes(self, text, lang=TTS_LANG, slow=TTS_SLOW):
        """
        Convert text to speech and return raw MP3 bytes
        
        Args:
            text: Text to convert to speech
            lang: Language code
            slow: Whether to speak slowly
        
        Returns:
            Bytes representing the MP3 audio file, or None if error
        """
        try:
            if not text or text.strip() == "":
                return None
            
            tts = gTTS(text=text, lang=lang, slow=slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            print(f"TTS Byte Generation Error: {e}")
            return None
    
    def stop(self):
        """Stop current speech"""
        if self.audio_available:
            pygame.mixer.music.stop()
    
    def is_speaking(self):
        """Check if currently speaking"""
        if self.audio_available:
            return pygame.mixer.music.get_busy()
        return False