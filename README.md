# Audio transcriber
The purpose of the script contained here is to transcribe an audio file (in the .wav format) to a .txt file. First, using [pyannote.audio](https://huggingface.co/pyannote), it will handle speaker diarization, identifying and segmenting the audio file into diferent fragments associated to a particular speaker. Then, using [Whisper](https://openai.com/index/whisper/), each audio fragment will be transcribed into a text file along with the respective speaker.

# Setup
Requirements:
- Create your OpenAI api key. You will need some credit in order to use Whisper's API.
- Accept [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) user conditions.
- Accept [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) user conditions.
- Create an access token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).

Then:
- Create .env file from .env.example and insert your API keys.
- Create your python virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Using the script
You can use the script in two different ways:

1. If you provide the path to a folder, every .wav file inside that folder will be transcribed: `python transcription.py <folder_path>`
2. If you provide the path to a .wav file, that file will be transcribed: `python transcription.py <file_path>`

A `resources/transcriptions` folder will be created and will be used to store every transcription.