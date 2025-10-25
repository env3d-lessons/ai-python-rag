from transformers import pipeline
# pipe = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

pipe2 = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]

while True:
    sentence = input("Message: ")
    if sentence.lower() == 'bye':
        break
    sentiment = pipe(sentence)[0]['label']
    
    pipe2_input =  [
        {"role": "user", "content": f"The user is feeling {sentiment}.  Make the user feel positive"}
    ]
        
    response = pipe2(pipe2_input, skip_prompt=True, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']    
    print("Response:", response)