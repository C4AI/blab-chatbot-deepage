from __future__ import annotations

import csv
from pathlib import Path
from sys import maxsize
from tempfile import TemporaryDirectory
from typing import Any, Callable

from datasets import DatasetDict, Dataset
from datasets.utils import disable_progress_bar
from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever, PreProcessor

from transformers import T5Tokenizer, IntervalStrategy
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from logging import getLogger


disable_progress_bar()

getLogger().setLevel("INFO")
logger = getLogger("deepage_bot")


class MessageType:
    """Represents a message type."""

    SYSTEM = "S"
    TEXT = "T"
    VOICE = "V"
    AUDIO = "a"
    VIDEO = "v"
    IMAGE = "i"
    ATTACHMENT = "A"


class DeepageBot:
    """A bot that usses Deepagé."""

    def __init__(
        self,
        model_dir: str | Path,
        idx_name: str,
        answer_function: Callable[[dict[str, Any]], None],
        k_retrieval: int,
        max_input_length: int = 1024,
        max_target_length: int = 32,
    ):
        self.k_retrieval = k_retrieval
        self.model_dir = model_dir
        self.idx_name = idx_name
        self.answer_function = answer_function
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

    def receive_message(self, message: dict[str, Any]) -> None:
        """Receive a message from the user or other bots.

        Messages from other bots are ignored.

        Args:
            message: the received message
        """
        if not message.get("sent_by_human", False):
            return
        question = message["text"]

        q = [
            {
                "question": [question],
                "answer": [""],
                "documents": self._find_relevant_documents(question)["documents"],
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
            self.answer_function({"type": MessageType.TEXT, "text": answer})

    def _find_relevant_documents(self, question: str):
        document_store = ElasticsearchDocumentStore(index=self.idx_name)
        retriever = ElasticsearchRetriever(document_store=document_store)
        extractive_pipeline = Pipeline()
        extractive_pipeline.add_node(
            component=retriever, name="ESRetriever1", inputs=["Query"]
        )
        return extractive_pipeline.run(
            query=question, params={"top_k": self.k_retrieval}
        )

    def _preprocess(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        questions = []
        answers = []
        for instance in docs:
            question = "question: " + instance["question"][0]
            doc = []
            for d in instance["documents"][: self.k_retrieval]:
                doc.append(d.content)
                question += "  context: " + d.meta["title"] + " " + d.content
            questions.append(question)
            answers.append(instance["answer"][0])
        return {"question": questions, "answer": answers}

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

    @classmethod
    def index(
        cls,
        document: str | Path,
        index_name: str,
        max_words: int,
        max_entries=maxsize,
    ) -> None:
        entries = []
        logger.info("reading document")
        with open(document, "r", encoding="utf-8") as fd:
            reader = csv.reader(fd, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                title, text = row
                entries.append({"content": text, "meta": {"title": title}})
                if len(entries) >= max_entries:
                    logger.info("stopping after %d entries" % max_entries)
                    break
        logger.info("finished reading document with %d entries" % len(entries))
        logger.info("pre-processing entries")
        # noinspection PyArgumentEqualDefault
        processor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by="word",
            split_length=max_words,
            split_respect_sentence_boundary=False,
            split_overlap=0,
        )
        docs = processor.process(entries)
        logger.info("opening Elasticsearch")
        document_store = ElasticsearchDocumentStore(index=index_name)
        existing = document_store.get_document_count()
        if existing:
            logger.info("deleting existing documents (%d)" % existing)
            document_store.delete_documents()
        logger.info("writing documents")
        document_store.write_documents(docs, batch_size=1000)
