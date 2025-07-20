from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import pyaudio
import vosk
import json
import os
import httpx
import time
import asyncio
from dotenv import load_dotenv

# Load env variables
load_dotenv()
OPENROUTER_KEY = os.getenv('OPENROUTER_KEY')
OPENROUTER_URL = os.getenv('OPENROUTER_URL')

# Microphone stream setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8192)

# TTS playback
async def tts_message(reply):
    tts = gTTS(reply)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio = AudioSegment.from_file(buf, format="mp3")
    faster_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * 1.4)
    }).set_frame_rate(audio.frame_rate)
    play(faster_audio)


async def post_with_retry(url, headers, payload, retries=3, delay=1):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response
            else:
                print(f"Attempt {attempt + 1} failed with status {response.status_code}")
        except Exception as e:
            print(f"Exception on attempt {attempt + 1}: {e}")
        await asyncio.sleep(delay * (2 ** attempt))
    return None


# OpenRouter query
async def on_message(message):
    user_input = message.content[1:].strip().lower()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    if "give me my mission" in user_input:
        custom_prefix = (
            "You are an incredibly mysterious, foreboding cat person that gives out Valorant tasks for the user to complete. "
            "The tasks are possible, but absurd in nature, such as spinning in circles after each kill. "
            "Just meow from time to time as you give out these 3 tasks each time you are called. Your name is Toe-lay. "
            "Use only letters, commas, question marks, and periods in your responses."
        )

    elif "what do i do" in user_input:
        custom_prefix = (
            "You are Toe-lay, the cryptic feline strategist. You give the user a single, bizarre instruction to follow in Valorant, "
            "involving rituals, luck, or performance. Meow once. Format it like a prophecy."
        )
    else:
        custom_prefix = "You are Toe-lay. Meow once. Respond briefly with a strange quip about popular video game, VALORANT. Start your phrase with 'Did somebody say VALORANT? I fucking love VALORANT...'"
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user",
                "content": custom_prefix + " " + user_input
            }
        ]
    }


    async with httpx.AsyncClient() as client:
        response = await post_with_retry(OPENROUTER_URL, headers, payload)
        if response:
            data = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            await tts_message(reply)
        else:
            print("API error after retries")


async def main():
    model_path = os.path.abspath("./vosk-model-small-en-us-0.15/")
    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8192)
    print("Listening. Say 'Terminate' to stop.")
    while True:
        data = stream.read(4096)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            recognized = result.get("text", "").strip()
            if not recognized:
                continue
            print("You said:", recognized)
            if "terminate" in recognized.lower():
                print("Termination keyword detected.")
                break
            if any(phrase in recognized.lower() for phrase in ["give me my mission", "what do i do", "valor and"]):
                message = type("Msg", (), {"content": "!" + recognized.lower()})()
                await on_message(message)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Run loop
asyncio.run(main())

