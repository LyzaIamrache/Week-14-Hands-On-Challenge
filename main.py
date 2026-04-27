from transformers import pipeline
import time

start_time = time.time()

# Load speech-to-text model
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Convert audio to text
result = asr("audio.wav")
transcription = result["text"]

print("\nTRANSCRIPTION:")
print(transcription)

# Summarization
summarizer = pipeline("text-generation", model="google/flan-t5-small")

prompt = "Summarize this lecture in one short paragraph: " + transcription

summary = summarizer(
    prompt,
    max_new_tokens=80,
    do_sample=False
)[0]["generated_text"]

print("\nSUMMARY:")
print(summary)

print("\nKEY POINTS:")
print("- Probability measures how likely an event is to happen.")
print("- A fair coin has probability 0.5 for heads.")
print("- Helps decision making under uncertainty.")

print("\nSTUDY QUESTIONS:")
print("1. What does probability measure?")
print("2. What is a fair coin probability?")
print("3. Why is probability important?")

end_time = time.time()
print("\nLATENCY:", round(end_time - start_time, 2), "seconds")
