from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load the trained model and tokenizer
model_name = "./trained_gelectra"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Initialize the pipeline
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Context and questions
context = """
The Roman Empire was a state that existed from 27 BC to AD 395, with its capital in Rome. It was one of the largest empires in history. 
The Roman Empire included vast territories in Europe, North Africa, and Western Asia. It was ruled by emperors and had a complex 
administrative structure. Rome was the central capital of the empire, and the Mediterranean Sea was the sea that surrounded the empire. 
The empire consisted of many provinces that were governed by Roman governors. Some of the most significant conquests included territories 
such as Gaul, Britain, Hispania, Greece, and Egypt.
"""

questions = [
    "What was the Roman Empire?",
    "Who ruled the Roman Empire?",
    "What territories did the Roman Empire include?",
    "What was the capital of the Roman Empire?",
    "What sea was surrounded by the Roman Empire?"
]

for question in questions:
    result = nlp(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print('-' * 50)
