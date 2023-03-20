"""DEEPAGÉ bot for BLAB."""

from __future__ import annotations

import csv
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any

from blab_chatbot_bot_client.conversation_websocket import (
    WebSocketBotClientConversation,
)
from blab_chatbot_bot_client.data_structures import (
    OutgoingMessage,
    Message,
    MessageType,
)
from datasets import Dataset, DatasetDict
from datasets.utils import disable_progress_bar
from haystack import Document, Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, PreProcessor
from overrides import overrides
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    IntervalStrategy,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
)

from blab_chatbot_deepage.deepage_settings_format import (
    BlabDeepageClientSettings,
    DeepageSettings,
)

disable_progress_bar()

getLogger().setLevel("INFO")
logger = getLogger("deepage_bot")


class DeepageBot(WebSocketBotClientConversation):
    """A bot that uses DEEPAGÉ."""

    settings: BlabDeepageClientSettings

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Create an instance.

        Args:
            args: positional arguments (passed to the parent class)
            kwargs: keyword arguments (passed to the parent class)
        """
        super().__init__(*args, **kwargs)
        model_dir = self.settings.DEEPAGE_SETTINGS["MODEL_PATH"]
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

    @overrides
    def generate_answer(self, message: Message) -> list[OutgoingMessage]:
        q = [
            {
                "question": [message.text],
                "documents": self._find_relevant_documents(message.text)["documents"],
            }
        ]
        dataset_test = Dataset.from_dict(self._preprocess(q))
        raw_datasets = DatasetDict({"test": dataset_test})
        tokenized_datasets = raw_datasets.map(
            lambda ex: self._preprocess_function(ex), batched=True
        )
        a = self.trainer.predict(
            tokenized_datasets["test"],
            max_length=self.settings.DEEPAGE_SETTINGS["MAX_TARGET_LENGTH"],
        )
        return [
            *map(
                lambda t: OutgoingMessage(
                    type=MessageType.TEXT,
                    text=t,
                    local_id=self.generate_local_id(),
                ),
                map(
                    lambda p: self.tokenizer.decode(p, skip_special_tokens=True),
                    a.predictions,
                ),
            )
        ]

    def _find_relevant_documents(self, question: str) -> list[dict[str, Document]]:
        document_store = ElasticsearchDocumentStore(
            index=self.settings.DEEPAGE_SETTINGS["ES_INDEX_NAME"]
        )
        retriever = BM25Retriever(document_store=document_store)
        extractive_pipeline = Pipeline()
        extractive_pipeline.add_node(
            component=retriever, name="ESRetriever1", inputs=["Query"]
        )
        return extractive_pipeline.run(
            query=question,
            params={"top_k": self.settings.DEEPAGE_SETTINGS["K_RETRIEVAL"]},
        )

    def _preprocess(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        questions = []
        answers = []
        for instance in docs:
            question = "question: " + instance["question"][0] + " context: "
            doc = []
            for d in instance["documents"][
                : self.settings.DEEPAGE_SETTINGS["K_RETRIEVAL"]
            ]:
                doc.append(d.content)
                question += " " + d.meta["title"] + " " + d.content
            questions.append(question)
            answers.append(instance.get("answer", [""])[0])
        return {"question": questions, "answer": answers}

    def _preprocess_input(self, examples: dict[str, Any]) -> dict[str, Any]:
        return self.tokenizer(
            examples["question"],
            max_length=self.settings.DEEPAGE_SETTINGS["MAX_INPUT_LENGTH"],
            truncation=True,
        )

    def _preprocess_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        model_inputs = self._preprocess_input(examples)
        # Set up the tokenizer for targets
        labels = self.tokenizer(
            examples["answer"],
            max_length=self.settings.DEEPAGE_SETTINGS["MAX_TARGET_LENGTH"],
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    @classmethod
    def index(cls, config: DeepageSettings, max_entries: int, max_words: int) -> None:
        """
        Index the entries in a document.

        If an old index exists, it is deleted.

        Args:
            config: settings for DEEPAGÉ
            max_entries: maximum number of entries to index
            max_words: maximum number of words per document
        """
        entries = []
        logger.info("reading document")
        with open(config["CSV_DOCUMENT_PATH"], encoding="utf-8") as fd:
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
        document_store = ElasticsearchDocumentStore(index=config["ES_INDEX_NAME"])
        existing = document_store.get_document_count()
        if existing:
            logger.info("deleting existing documents (%d)" % existing)
            document_store.delete_documents()
        logger.info("writing documents")
        document_store.write_documents(docs, batch_size=1000)
