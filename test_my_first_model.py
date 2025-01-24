from transformers import pipeline

model_id = "myfirstmodel"
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
)
messages = [
    {"role": "user", "content": "How many legs are there on a farm, if there are 3 cows, 2 chickens, and 1 pig?"},
]
outputs = pipe(
    messages,
    max_new_tokens=128
)
print(outputs[0]["generated_text"][-1])
