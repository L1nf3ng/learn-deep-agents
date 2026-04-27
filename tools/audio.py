import os
import uuid
from pathlib import Path
import requests


MINIMAX_T2A_URL = "https://api.minimaxi.com/v1/t2a_v2"


def text_to_speech(
    text: str,
    model: str = "speech-2.8-hd",
    voice_id: str = "female-yujie",
    speed: int = 1,
    vol: int = 1,
    pitch: int = 0,
    emotion: str = "happy",
    sample_rate: int = 32000,
    bitrate: int = 128000,
    audio_format: str = "mp3",
    channel: int = 1,
    pronunciation_dict: list = None,
    text_normalization: bool = False,
    force_cbr: bool = False,
    aigc_watermark: bool = False,
    output_type: str = "hex",
    latex_read: bool = False,
) -> str:
    """Convert text to speech using MiniMax T2A API and save audio to local file.

    Args:
        text: Text to synthesize. Max 10000 chars.
        model: TTS model. Options: speech-2.8-hd, speech-2.8-turbo, speech-2.6-hd,
            speech-2.6-turbo, speech-02-hd, speech-02-turbo, speech-01-hd, speech-01-turbo.
        voice_id: Voice ID. Default: Chinese female voice.
        speed: Speech speed, 1-10 (default: 1).
        vol: Volume, 1-10 (default: 1).
        pitch: Pitch adjustment, [-12, 12] (default: 0).
        emotion: Emotion setting. Options: happy, sad, angry, fearful, disgusted,
            surprised, calm, fluent, whisper. Note: whisper not supported by speech-2.8-*.
        sample_rate: Audio sample rate in Hz (default: 32000).
        bitrate: Audio bitrate in bps (default: 128000).
        audio_format: Audio format: mp3, wav, flac (default: mp3).
        channel: Audio channel, 1=mono, 2=stereo (default: 1).
        pronunciation_dict: Pronunciation overrides, e.g. ["词/(ci2)"].
        text_normalization: Enable CN/EN text normalization (default: False).
        force_cbr: Force constant bitrate, only for streaming mp3 (default: False).
        aigc_watermark: Add audio watermark at end (default: False).
        output_type: "hex" or "url", default "hex".
        latex_read: Enable LaTeX reading, only for Chinese (default: False).

    Returns:
        Path to the saved audio file.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    voice_setting = {
        "voice_id": voice_id,
        "speed": speed,
        "vol": vol,
        "pitch": pitch,
        "emotion": emotion,
    }
    if text_normalization:
        voice_setting["text_normalization"] = text_normalization
    if latex_read:
        voice_setting["latex_read"] = latex_read

    audio_setting = {
        "sample_rate": sample_rate,
        "bitrate": bitrate,
        "format": audio_format,
        "channel": channel,
    }
    if force_cbr:
        audio_setting["force_cbr"] = force_cbr

    payload = {
        "model": model,
        "text": text,
        "stream": False,
        "voice_setting": voice_setting,
        "audio_setting": audio_setting,
        "output_type": output_type,
    }
    if pronunciation_dict:
        payload["pronunciation_dict"] = {"tone": pronunciation_dict}
    if aigc_watermark:
        payload["aigc_watermark"] = aigc_watermark

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post( MINIMAX_T2A_URL, json=payload, headers=headers, timeout=60 )
    response.raise_for_status()

    result = response.json()

    audio_hex = result.get("data", {}).get("audio", "")
    if not audio_hex:
        raise ValueError(f"No audio in response: {result}")
    audio_bytes = bytes.fromhex(audio_hex)

    output_dir = Path(__file__).parent.parent / "results"

    safe_name = (
        text[:30]
        .translate(str.maketrans("", "", "\n\r\t"))
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )
    filename = f"tts_{safe_name}_{uuid.uuid4().hex[:6]}.{audio_format}"
    filepath = output_dir / filename

    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    return str(filepath)
