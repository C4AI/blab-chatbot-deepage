from datasets import DatasetDict, Dataset
from transformers import T5Tokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


all_questions = [
    {
        "question": [
            "Quantas das florestas da Amazônia são ameaçadas pela mudança climática"
        ],
        "answer": [""],
        "documents": [],
    },
]


model_checkpoint = "model/QA-ptt5-base-wiki-k10/checkpoint-9000"
K_Retrieval = 10
max_input_length = 1024
max_target_length = 32


model_name = "QA-ptt5-base-wiki-k10"


test_docs = all_questions


def preprocess(docs):
    questions = []
    answer = []
    docsReturn = []
    for instance in docs:
        question = "question: " + instance["question"][0]
        doc = []
        for i in range(min(K_Retrieval, len(instance["documents"]))):
            document_dict = {**instance["documents"][i]}
            document = document_dict["meta"]["title"] + " " + document_dict["text"]
            doc.append(document_dict["text"])
            question += "  context: " + document
        questions.append(question)
        answer.append(instance["answer"][0])
        docsReturn.append(doc)

    pd_dataset = {"question": questions, "answer": answer}
    return (pd_dataset, docsReturn)


pd_dataset_test, docs = preprocess(test_docs)


dataset_test = Dataset.from_dict(pd_dataset_test)


raw_datasets = DatasetDict({"test": dataset_test})

tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)


def preprocess_input(examples):
    model_inputs = tokenizer(
        examples["question"], max_length=max_input_length, truncation=True
    )
    return model_inputs


def preprocess_function(examples):
    model_inputs = preprocess_input(examples)
    # Setup the tokenizer for targets
    labels = tokenizer(
        examples["answer"], max_length=max_target_length, truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 4
args = Seq2SeqTrainingArguments(
    model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=30,
    predict_with_generate=True,
    # fp16=True,  # apparently this requires a GPU
    gradient_accumulation_steps=4,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    args,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
a = trainer.predict(tokenized_datasets["test"], max_length=max_target_length)

answers = []
test_labels = []
for i in range(len(a.predictions)):
    answers.append(tokenizer.decode(a.predictions[i], skip_special_tokens=True))


print(answers)
