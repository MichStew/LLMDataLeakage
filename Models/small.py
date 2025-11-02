# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("katanemo/Arch-Router-1.5B")
model = AutoModelForCausalLM.from_pretrained("katanemo/Arch-Router-1.5B")

torch.cuda.set_device=("gpu")

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages, # change this to a sensitive dataset 
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device) # run this at home 

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
