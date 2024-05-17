from __future__ import annotations
import time
import os
import shutil
import sys
import soundfile as sf
from typing import Any
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def replace_speakers_names(transcribed_file_path: str) -> None:
    with open(transcribed_file_path, "r") as file:
        conversation = file.read()

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        "Read the following conversation: {conversation}."
        "Find the names of the speakers in the text. Replace SPEAKER_XX by their respective names."
        "Do not make up names for them. Do not add an introduction of any kind, like 'Sure, here is the conversation with the names of the speakers replaced:'"
        "Follow this template:"
        "ALEX\nHello, this is Alex.\n\n"
        "MATT\nAnd this is Matt!\n\n"
    )
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    answer = chain.invoke({"conversation": conversation})

    # Overwrite original file with answer
    with open(transcribed_file_path, "w") as output_file:
        output_file.write(answer)
        print(f"Transcription saved as {transcribed_file_path}")


def cleanup(diarizations_path: str) -> None:
    try:
        shutil.rmtree(diarizations_path)
    except OSError as e:
        print(f"Error: {diarizations_path} : {e.strerror}")


def transcribe_audio(
    raw_audio_file_path: str,
    transcribed_file_path: str,
    speaker_id: str,
    openai_apikey: str,
) -> None:
    client = OpenAI(api_key=openai_apikey)

    with open(raw_audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )

        # Create the transcription folder if it doesn't exist
        output_folder = os.path.dirname(transcribed_file_path)
        os.makedirs(output_folder, exist_ok=True)

        # Append speaker ID and transcription to the output txt file
        with open(transcribed_file_path, "a") as output_file:
            output_file.write(f"{speaker_id}\n")
            output_file.write(transcription.text)
            output_file.write("\n\n")


def get_audio_duration(audio_file_path: str) -> float:
    with sf.SoundFile(audio_file_path, "r") as f:
        return len(f) / f.samplerate


def extract_chunk_number(file_name: str) -> int:
    return int(file_name.split("-chunk_")[1].split("-")[0])


def transcribe_diarized_files(
    diarizations_path: str, transcribed_file_path: str, openai_key: str
) -> None:
    file_names_sorted = sorted(os.listdir(diarizations_path), key=extract_chunk_number)

    for file_name in file_names_sorted:
        try:
            if file_name.endswith(".wav"):
                raw_audio_file_path = os.path.join(diarizations_path, file_name)
                duration = get_audio_duration(raw_audio_file_path)

                # A file needs to be at least 0.1 seconds long to be transcribed by Whisper.
                if duration >= 0.1:
                    speaker_id = file_name.split("-speaker_")[-1].split(".")[0]
                    transcribe_audio(
                        raw_audio_file_path,
                        transcribed_file_path,
                        speaker_id,
                        openai_key,
                    )
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue


# Save each segment associated to a unique speaker as separate audio file
def diarize_audio(diarizations_path: str, audio_file: str, diarization: Any) -> None:
    audio = AudioSegment.from_wav(audio_file)

    # Specify output directory and create it if it doesn't exist
    os.makedirs(diarizations_path, exist_ok=True)

    for i, (segment, section_id, speaker_id) in enumerate(diarization):
        start_time, end_time = segment.start, segment.end

        # Extract the segment from the original audio
        segment_audio = audio[start_time * 1000 : end_time * 1000]

        # Save the segment as a separate audio file
        output_file = os.path.join(
            diarizations_path,
            f"{os.path.splitext(os.path.basename(audio_file))[0]}-chunk_{i+1}-speaker_{speaker_id}.wav",
        )
        segment_audio.export(output_file, format="wav")
        print(f"Fragment {i+1} saved as {output_file}")


def get_required_env_variable(env_var_name: str) -> str:
    value = os.getenv(env_var_name)
    if value is None:
        raise ValueError(f"Environment variable {env_var_name} is not set.")
    return value


def diarize_and_transcribe(raw_audio_file_path: str) -> None:
    load_dotenv()

    start_time = time.perf_counter()

    # Set paths to directories
    base_path = os.path.dirname(__file__)
    resources_path = os.path.join(base_path, "resources")
    diarizations_path = os.path.join(resources_path, "diarizations")
    transcriptions_path = os.path.join(resources_path, "transcriptions")

    # Set paths to files
    base_file_name = os.path.basename(raw_audio_file_path)
    base_name_without_extension = os.path.splitext(base_file_name)[0]
    transcribed_file_path = os.path.join(
        transcriptions_path, base_name_without_extension + ".txt"
    )

    # Fetch api keys from environment
    openai_key = get_required_env_variable("OPENAI_API_KEY")
    huggingface_key = get_required_env_variable("HUGGINGFACE_API_KEY")

    # Execute speaker diarization on audio file
    print(f"Starting diarization for {base_file_name}...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=huggingface_key,
    )
    diarization = pipeline(raw_audio_file_path)
    print(f"Finished diarization for {base_file_name}!")

    # Save segment associated to each speaker as a separate audio file
    print(f"Starting file fragmentation for {base_file_name}...")
    diarize_audio(
        diarizations_path,
        raw_audio_file_path,
        diarization.itertracks(yield_label=True),
    )
    print(f"Finished file fragmentation for {base_file_name}!")

    # Transcribe audio segments into unified text file
    print(f"Starting transcription for {base_file_name}...")
    transcribe_diarized_files(diarizations_path, transcribed_file_path, openai_key)
    print(f"Finished transcription for {base_file_name}!")

    # Remove diarizations folder and its files
    cleanup(diarizations_path)

    # Start name replacement
    print(f"Starting name replacement for {base_file_name}...")
    replace_speakers_names(transcribed_file_path)
    print(f"Finished name replacement for {base_file_name}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time:", round(elapsed_time, 2), "seconds!\n")


def main():
    if len(sys.argv) != 2:
        print(
            "Error: You need to provided a path as an argument.\n"
            "Usage:\n"
            "- python transcription.py <folder_path> (will transcribe every .wav file within the folder)\n"
            "- python transcription.py <file_path> (will transcribe a specific .wav file)"
        )
        sys.exit(1)

    base_path = os.getcwd()
    relative_audio_path = sys.argv[1]
    absolute_audio_path = os.path.join(base_path, relative_audio_path)
    if os.path.isdir(absolute_audio_path):
        for file_name in sorted(os.listdir(absolute_audio_path)):
            try:
                if file_name.endswith(".wav"):
                    file_path = os.path.join(absolute_audio_path, file_name)
                    diarize_and_transcribe(file_path)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}\n")
                continue

    elif absolute_audio_path.lower().endswith(".wav"):
        diarize_and_transcribe(absolute_audio_path)

    else:
        print("Error: The provided path is neither a directory nor a .wav file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
