"""DEEPAGÉ bot for BLAB."""

from __future__ import annotations

import csv
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, cast

from blab_chatbot_bot_client.conversation_websocket import (
    WebSocketBotClientConversation,
)
from blab_chatbot_bot_client.data_structures import (
    Message,
    MessageType,
    OutgoingMessage,
)
from datasets import Dataset, DatasetDict
from datasets.utils import disable_progress_bar
from haystack import Document, Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, PreProcessor
from overrides import overrides
from transformers import (
    AutoModelForSeq2SeqLM,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    IntervalStrategy,
    PreTrainedModel,
    PreTrainedTokenizer,
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


class DeepageBot(WebSocketBotClientConversation[BlabDeepageClientSettings]):
    """A bot that uses DEEPAGÉ."""

    @classmethod
    @lru_cache
    def _load_model(
        cls, model_dir: str
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel, Seq2SeqTrainer]:
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

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

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model,
            args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        return tokenizer, model, trainer

    def __init__(self, *args: Any, **kwargs: Any):
        """Create an instance.

        Args:
            args: positional arguments (passed to the parent class)
            kwargs: keyword arguments (passed to the parent class)
        """
        super().__init__(*args, **kwargs)
        model_dir = self.settings.DEEPAGE_SETTINGS["MODEL_PATH"]
        self.tokenizer, self.model, self.trainer = self._load_model(model_dir)

    @overrides
    def on_receive_message(self, message: Message) -> None:
        if message.sent_by_human and message.type == MessageType.TEXT:
            for answer in self.generate_answer(message):
                self.enqueue_message(answer)

    @overrides
    def generate_answer(self, message: Message) -> list[OutgoingMessage]:
        if not message.text:
            return []
        q = [
            {
                "question": [message.text],
                "documents": self._find_relevant_documents(message.text),
            }
        ]
        dataset_test = Dataset.from_dict(self._preprocess(q))
        raw_datasets = DatasetDict({"test": dataset_test})
        tokenized_datasets = raw_datasets.map(
            lambda ex: self._preprocess_function(ex), batched=True
        )
        predictions = self.trainer.predict(
            tokenized_datasets["test"],
            max_length=self.settings.DEEPAGE_SETTINGS["MAX_TARGET_LENGTH"],
        ).predictions
        decoded_predictions = [
            self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions
        ]
        return [
            OutgoingMessage(
                type=MessageType.TEXT,
                text=t,
                local_id=self.generate_local_id(),
                quoted_message_id=message.id,
            )
            for t in decoded_predictions
        ]

    def _find_relevant_documents(self, question: str) -> list[Document]:
        document_store = ElasticsearchDocumentStore(
            index=self.settings.DEEPAGE_SETTINGS["ES_INDEX_NAME"]
        )
        retriever = BM25Retriever(document_store=document_store)
        extractive_pipeline = Pipeline()
        extractive_pipeline.add_node(
            component=retriever, name="ESRetriever1", inputs=["Query"]
        )
        return cast(
            List[Document],
            extractive_pipeline.run(
                query=question,
                params={"top_k": self.settings.DEEPAGE_SETTINGS["K_RETRIEVAL"]},
            )["documents"],
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

    def _preprocess_input(self, examples: dict[str, Any]) -> BatchEncoding:
        return cast(
            BatchEncoding,
            self.tokenizer(
                examples["question"],
                max_length=self.settings.DEEPAGE_SETTINGS["MAX_INPUT_LENGTH"],
                truncation=True,
            ),
        )

    def _preprocess_function(self, examples: dict[str, Any]) -> BatchEncoding:
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
        """Index the entries in a document.

        If an old index exists, it is deleted.

        Args:
            config: settings for DEEPAGÉ
            max_entries: maximum number of entries to index
            max_words: maximum number of words per document
        """
        entries = []
        logger.info("reading document")
        with Path(config["CSV_DOCUMENT_PATH"]).open(encoding="utf-8") as fd:
            reader = csv.reader(fd, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                title, text = row
                entries.append({"content": text, "meta": {"title": title}})
                if len(entries) >= max_entries:
                    logger.info("stopping after {n} entries", extra={"n": max_entries})
                    break
        logger.info(
            "finished reading document with {n} entries",
            extra={"n": len(entries)},
        )
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
            logger.info("deleting existing documents ({n})", extra={"n": existing})
            document_store.delete_documents()
        logger.info("writing documents")
        document_store.write_documents(docs, batch_size=1000)
