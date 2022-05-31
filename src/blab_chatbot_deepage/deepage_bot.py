from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple, Callable

from datasets import DatasetDict, Dataset
from datasets.utils import disable_progress_bar

from transformers import T5Tokenizer, IntervalStrategy
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from .controller_interface import ConversationInfo, Message, MessageType

disable_progress_bar()


class DeepageBot:
    """A bot that usses Deepagé."""

    def __init__(
        self,
        conversation_info: ConversationInfo,
        *,
        model_dir: str | Path,
        k_retrieval: int,
        max_input_length: int = 1024,
        max_target_length: int = 32,
    ):
        self.k_retrieval = k_retrieval
        self.conversation_info = conversation_info
        self.model_dir = model_dir
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        output_dir = TemporaryDirectory().name
        # apparently this is not used in this case, but the argument is required

        args = Seq2SeqTrainingArguments(
            output_dir,
            evaluation_strategy=IntervalStrategy.EPOCH,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=30,
            predict_with_generate=True,
            gradient_accumulation_steps=4,
            disable_tqdm=True,
            log_level="warning",
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.trainer = Seq2SeqTrainer(
            self.model,
            args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def receive_message(self, message: Message) -> None:
        """Receive a message from the user or other bots.

        Messages from other bots are ignored.

        Args:
            message: the received message
        """
        if not message.sent_by_human():
            return
        q = [
            {
                "question": [message.text],
                "answer": [""],
                "documents": [],
            },
        ]
        dataset_test = Dataset.from_dict(self._preprocess(q))
        raw_datasets = DatasetDict({"test": dataset_test})
        tokenized_datasets = raw_datasets.map(
            lambda ex: self._preprocess_function(ex), batched=True
        )
        a = self.trainer.predict(
            tokenized_datasets["test"], max_length=self.max_target_length
        )
        for prediction in a.predictions:
            answer = self.tokenizer.decode(prediction, skip_special_tokens=True)
            self.conversation_info.send_function(
                {"type": MessageType.TEXT, "text": answer}
            )

    def _preprocess(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        questions = []
        answer = []
        for instance in docs:
            question = "question: " + instance["question"][0]
            doc = []
            for _i in range(min(self.k_retrieval, len(instance["documents"]))):
                document_dict = {**instance["documents"][i]}
                document = document_dict["meta"]["title"] + " " + document_dict["text"]
                doc.append(document_dict["text"])
                question += "  context: " + document
            questions.append(question)
            answer.append(instance["answer"][0])
        return {"question": questions, "answer": answer}

    def _preprocess_input(self, examples):
        return self.tokenizer(
            examples["question"], max_length=self.max_input_length, truncation=True
        )

    def _preprocess_function(self, examples):
        model_inputs = self._preprocess_input(examples)
        # Setup the tokenizer for targets
        labels = self.tokenizer(
            examples["answer"], max_length=self.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
