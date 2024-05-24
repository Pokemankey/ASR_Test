import whisper
import time
import csv
import os
from pydub import AudioSegment
import torch

gpuname = "3090"
modelname = "small"
def get_audio_length(audio_file):
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000  # Length in seconds

# Function to perform inference on audio files and calculate inference time
def perform_inference(model, audio_files, device):
    results = []
    i = 1
    for audio_file in audio_files:
        print(i)
        start_time = time.time()
        audio_length = get_audio_length(audio_file)
        with torch.cuda.device(device):
            result = model.transcribe(audio_file)
        # result = model.transcribe(audio_file)
        end_time = time.time()
        inference_time = end_time - start_time
        results.append((audio_file, audio_length, inference_time))
        i += 1
    return results

# Function to write results to CSV file
def write_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Audio File', 'Audio Length (s)', 'Inference Time (s)'])
        for result in results:
            # Encode each string in the result tuple to UTF-8
            writer.writerow([str(item).encode('utf-8') if isinstance(item, str) else item for item in result])

if __name__ == "__main__":
    # Load the Whisper model
    model = whisper.load_model(modelname)

    torch.cuda.init()
    device = "cuda"

    # Directory containing audio files
    # audio_directory = r"D:\Repos\thebox4_Stresstest\audio_dataset"
    audio_directory = "./audio_dataset"


    # List audio files
    audio_files = [os.path.join(audio_directory, file) for file in os.listdir(audio_directory) if file.endswith(".mp3")]

    # Perform inference and calculate inference time
    results = perform_inference(model, audio_files, device)

    # Output CSV file
    output_file = gpuname+"_"+modelname+"_benchmark.csv"

    # Write results to CSV
    write_to_csv(results, output_file)

    print("Inference results saved to:", output_file)
