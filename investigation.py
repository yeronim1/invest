from transformers import AutoTokenizer, AutoModelForQuestionAnswering, GPT2Tokenizer, GPT2LMHeadModel
import torch
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Завантаження стоп-слів
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Підключення до MongoDB
client = MongoClient('localhost', 27017)
db = client.qa_system
contexts_collection = db.contexts

# Завантаження токенізатора і моделі для відповіді на питання
qa_model_name = "./trained_gelectra"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Завантаження токенізатора і моделі GPT-2 для генерації повних відповідей
gpt_model_name = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

def find_best_context(question):
    processed_question = preprocess_text(question)
    query = {"$text": {"$search": " ".join(processed_question)}}
    count = contexts_collection.count_documents(query)
    if count > 0:
        result = contexts_collection.find(query).sort([("score", {"$meta": "textScore"})]).limit(1)
        context = result[0]["text"]
        return context
    else:
        return None

def save_model_and_tokenizer(model, tokenizer, model_dir):
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def generate_full_answer(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer: "
    inputs = gpt_tokenizer(prompt, return_tensors='pt')
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = gpt_model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=500,
            num_return_sequences=1,
            pad_token_id=gpt_tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control the randomness of predictions
            top_p=0.9,        # Nucleus sampling
            top_k=50          # Top-k sampling
        )
    full_answer = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_answer.split("Answer: ")[-1].strip()

def answer_question(question):
    context = find_best_context(question)
    if context:
        inputs = qa_tokenizer(question, context, return_tensors='pt')

        # Передбачення короткої відповіді
        with torch.no_grad():
            outputs = qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            short_answer = qa_tokenizer.convert_tokens_to_string(
                qa_tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))

        # Оновлення вагів моделі
        start_positions = torch.tensor([answer_start])
        end_positions = torch.tensor([answer_end])
        outputs = qa_model(**inputs, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Збереження моделі
        save_model_and_tokenizer(qa_model, qa_tokenizer, qa_model_name)

        # Генерація повної відповіді з використанням GPT-2 без включення "Short-answer"
        full_answer = generate_full_answer(question, context + " " + short_answer)

        return full_answer
    else:
        return "Context not found."

optimizer = torch.optim.AdamW(qa_model.parameters(), lr=5e-5)

user_question = input('User input: ')

answer = answer_question(user_question)
print(f"Question: {user_question}\nAnswer: {answer}")
