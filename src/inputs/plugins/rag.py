import json
import asyncio
import logging
import uuid
from queue import Empty, Queue
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from inputs.base import SensorConfig
from inputs.base.loop import FuserInput


class RagInput(FuserInput[str]):
    """RAG input handler for OpenMind O1"""

    def __init__(self, config: Optional[SensorConfig] = None):
        super().__init__(config or SensorConfig())
        
        self.buffer: List[str] = []
        self.message_buffer: Queue[str] = Queue()
        self.context: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Configuration parameters
        self.doc_directory = getattr(config, "data_path", "./data")
        self.embedding_model = getattr(config, "embedding_model", "text-embedding-3-large")
        self.api_key = getattr(config, "openai_api_key")
        self.chunk_size = getattr(config, "chunk_size")
        self.chunk_overlap = getattr(config, "chunk_overlap")
        
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model, api_key=self.api_key)
        self.vector_store = self._initialize_vector_store()

    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _init_session(self):
        """Initialize aiohttp session if not exists."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)

    def _get_data_files(self) -> List[str]:
        """Retrieve a list of all relevant PDF, TXT, MD files in the data folder."""
        import glob
        files = []
        for pattern in ["**/*.pdf", "**/*.txt", "**/*.md"]:
            matched = glob.glob(os.path.join(self.doc_directory, pattern), recursive=True)
            # Exclude anything in faiss_index folder if needed
            matched = [f for f in matched if "faiss_index" not in f]
            files.extend(matched)
        return sorted(files)

    def _initialize_vector_store(self):
        """
        Initialize or load the FAISS vector store by *only* checking if any
        PDF/TXT/MD filenames have been added or removed since last time.
        """
        checkpoint_path = os.path.join(self.doc_directory, 'faiss_index_checkpoint.json')
        index_path = os.path.join(self.doc_directory, 'faiss_index')

        # Grab the list of PDF, TXT, MD files *currently* in the folder.
        current_file_list = self._get_data_files()

        # If index + checkpoint exist, try to load them
        if os.path.exists(index_path) and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    saved_file_list = json.load(f)

                if saved_file_list == current_file_list:
                    logging.info("No new or removed files; loading existing FAISS index.")
                    vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                    return vector_store
                else:
                    logging.info("Detected added or removed files; rebuilding the FAISS index.")
            except Exception as e:
                logging.warning("Error loading checkpoint; rebuilding FAISS index: %s", e)
        else:
            logging.info("No existing index/checkpoint found; building new FAISS index.")

        # Build a new FAISS index
        loader = DirectoryLoader(self.doc_directory, glob=["**/*.pdf", "**/*.txt", "**/*.md"])
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)

        # Create a FAISS index
        dummy_vector = self.embeddings.embed_query("dimension_test")
        dimension = len(dummy_vector)
        index = faiss.IndexFlatL2(dimension)

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Add documents with generated UUIDs
        uuids = [str(uuid.uuid4()) for _ in range(len(all_splits))]
        vector_store.add_documents(documents=all_splits, ids=uuids)
        vector_store.save_local(index_path)
        logging.info("Created and saved new FAISS index at %s.", index_path)

        # Overwrite the checkpoint file with the new list of files
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(current_file_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Failed to save checkpoint: %s", e)

        return vector_store

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages in the buffer.

        Returns
        -------
        Optional[str]
            Message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.5)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def raw_to_text(self, raw_input: Optional[str] = None):
        """Convert raw input to text format and add to buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to process. If None, process from message buffer.

        Returns
        -------
        str
            The processed text
        """
        if raw_input:
            self.message_buffer.put_nowait(raw_input)

        if self.message_buffer:
            try:
                message = self.message_buffer.get_nowait()
                if message:
                    self.buffer.append(message)
                    logging.debug(f"Added to buffer: {message}")
                    return message
            except Empty:
                pass

        return ""
    
    async def _raw_to_text(self, raw_input: Optional[str] = None):
        """Convert raw input to text format and add to buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to process. If None, process from message buffer.

        Returns
        -------
        str
            The processed text
        """
        if raw_input:
            self.message_buffer.put_nowait(raw_input)

        if self.message_buffer:
            try:
                message = self.message_buffer.get_nowait()
                if message:
                    self.buffer.append(message)
                    logging.debug(f"Added to buffer: {message}")
                    return message
            except Empty:
                pass

        return ""

    def formatted_latest_buffer(self) -> Optional[str]:
            """Format and return the context."""
            content = (
                self.context if self.context else (self.buffer[-1] if self.buffer else None)
            )

            if not content:
                return None

            result = f"""
            RagInput CONTEXT
            // START
            {content}
            // END
            """

            return result

