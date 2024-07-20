from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset

# Завантаження датасету SQuAD
dataset = load_dataset("squad")

# Скорочення датасету до 1% від оригінального розміру
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * 0.01)))
small_eval_dataset = dataset["validation"].shuffle(seed=42).select(range(int(len(dataset["validation"]) * 0.01)))

# Завантаження токенізатора і моделі
model_name = "deepset/gelectra-base-germanquad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Токенізація датасету
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Пошук позицій початку і кінця відповіді
        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(next((idx for idx, (start, end) in enumerate(offsets) if start <= start_char < end), 0))
            end_positions.append(next((idx for idx, (start, end) in enumerate(offsets) if start < end_char <= end), 0))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True, remove_columns=small_train_dataset.column_names)
tokenized_eval_dataset = small_eval_dataset.map(preprocess_function, batched=True, remove_columns=small_eval_dataset.column_names)

# Налаштування параметрів тренування
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Конфігурація тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Тренування моделі
trainer.train()

# Збереження натренованої моделі
model.save_pretrained("./trained_gelectra")
tokenizer.save_pretrained("./trained_gelectra")
