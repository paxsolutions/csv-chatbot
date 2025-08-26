import os
import tempfile
import pandas as pd
import hashlib
import time
import random
import logging
import numpy as np
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st


cwd = os.getcwd()
log_dir = f'{cwd}/logs'

os.makedirs(log_dir, exist_ok=True)

# Create log files if they don't exist
log_files = [
    'faiss_test.log',
    'faiss_error.log',
    'debug_error.log',
    'debug_dimensions.log',
    'debug_dimension_error.log'
]

for log_file in log_files:
    log_path = f'{log_dir}/{log_file}'
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/embedder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def dependable_faiss_import():
    """Import FAISS with fallback handling"""
    try:
        import faiss
        return faiss
    except ImportError:
        st.error("FAISS not installed. Please install with: pip install faiss-cpu")
        raise ImportError("FAISS library not found")


class Embedder:
    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def _get_file_hash(self, file_content):
        """
        Generate a hash for file content to enable caching
        """
        return hashlib.md5(file_content).hexdigest()

    def _sample_large_csv(self, df, max_rows=5000):
        """
        Sample large CSV files to reduce processing time while maintaining representativeness
        """
        if len(df) <= max_rows:
            return df

        # Take a stratified sample if possible, otherwise random sample
        try:
            # Try to sample from different parts of the dataset
            step = len(df) // max_rows
            sampled_df = df.iloc[::step][:max_rows]
            st.info(f"Large file detected ({len(df):,} rows). Using {len(sampled_df):,} representative rows for faster processing.")
            return sampled_df
        except:
            # Fallback to random sampling
            sampled_df = df.sample(n=max_rows, random_state=42)
            st.info(f"Large file detected ({len(df):,} rows). Using {len(sampled_df):,} random rows for faster processing.")
            return sampled_df

    def getDocEmbeds(self, file, filename):
        """
        Get document embeddings for a file - wrapper around storeDocEmbeds
        """
        return self.storeDocEmbeds(file, filename)

    def storeDocEmbeds(self, file, filename):
        """
        Stores document embeddings using optimized processing for large files
        """
        # Log entry point
        logger.info(f"Starting storeDocEmbeds for file: {filename}")
        with open(f'{log_dir}/embedder.log', 'a') as f:
            f.write(f"ENTRY: storeDocEmbeds called for {filename}\n")

        try:
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: About to generate file hash\n")

            # Caching
            file_hash = self._get_file_hash(file)

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: File hash generated: {file_hash[:10]}...\n")

            cache_filename = f"{filename}_{file_hash}"

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Checking for cached directory: {cache_filename}\n")

            # Check if cached version exists (FAISS format)
            cache_path = f"{self.PATH}/{cache_filename}"
            if os.path.isdir(cache_path):
                st.info("Using cached embeddings for faster loading...")
                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"STEP: Found cached directory, loading...\n")
                try:
                    embeddings = OpenAIEmbeddings()
                    cached_vectors = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"SUCCESS: Cached vectors loaded successfully\n")
                    return cached_vectors
                except Exception as e:
                    error_msg = f"Cached vectors corrupted: {str(e)}"
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"ERROR: {error_msg}\n")
                        f.write(f"STEP: Removing corrupted cache and regenerating\n")
                    st.warning("Cached vectors corrupted, regenerating embeddings...")
                    # Remove corrupted cache directory
                    try:
                        import shutil
                        shutil.rmtree(cache_path)
                    except:
                        pass

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: No cached file found, proceeding with processing\n")

        except Exception as e:
            error_msg = f"Error in storeDocEmbeds caching: {str(e)}"
            logger.error(error_msg)
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            st.error(error_msg)
            return None

        # Write the uploaded file to a temp file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        try:
            # Log progress
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Reading CSV file {filename}\n")

            # Read CSV with pandas and optimize for large files
            df = pd.read_csv(tmp_file_path, encoding="utf-8")
            df = self._sample_large_csv(df)

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: CSV loaded, shape: {df.shape}\n")

            if len(df) > 500:
                documents = self._process_large_csv_optimized(df)
            else:
                # Use standard CSVLoader for smaller files
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
                documents = loader.load_and_split()

            # Filter out empty documents before processing
            documents = [doc for doc in documents if doc.page_content.strip()]

            if not documents:
                st.error("No valid documents found in CSV file.")
                return None

            # Text splitting - optimized for numerical data preservation
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Larger chunks to preserve more context for numerical analysis
                chunk_overlap=400,  # Higher overlap to ensure numerical data isn't split
                length_function=len,
            )

            # Split documents
            split_docs = []
            for doc in documents:
                if doc.page_content.strip():  # Only process non-empty documents
                    chunks = text_splitter.split_documents([doc])
                    # Ensure all chunks have content
                    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
                    split_docs.extend(valid_chunks)

            if not split_docs:
                st.error("No valid document chunks created.")
                return None

            # Create embeddings
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Creating OpenAI embeddings instance\n")

            embeddings = OpenAIEmbeddings()

            # Process embeddings in single batch to avoid dimension issues
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: Calling _create_embeddings_single_batch with {len(split_docs)} documents\n")

            all_vectors = self._create_embeddings_single_batch(split_docs, embeddings)

            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"STEP: _create_embeddings_single_batch returned: {type(all_vectors)}\n")

            # Save cache using FAISS's built-in save method
            try:
                cache_path = f"{self.PATH}/{cache_filename}"
                all_vectors.save_local(cache_path)

                # Also save with original filename
                original_path = f"{self.PATH}/{filename.replace('.csv', '')}"
                all_vectors.save_local(original_path)

                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"SUCCESS: Saved vectors to cache using FAISS save_local\n")
            except Exception as e:
                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"WARNING: Failed to save cache: {str(e)}\n")
                # Continue without caching if save fails

            # Return vectors
            return all_vectors

        finally:
            os.remove(tmp_file_path)

    def _process_large_csv_optimized(self, df):
        """
        Process CSV files with comprehensive data coverage for accurate numerical analysis
        """
        documents = []

        # For small-medium datasets, create one comprehensive document
        if len(df) <= 100:
            full_text = f"Complete CSV Dataset:\n"
            full_text += f"Columns: {', '.join(df.columns)}\n"
            full_text += f"Total Rows: {len(df)}\n\n"

            # Add all rows with clear formatting
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}: "
                row_items = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        row_items.append(f"{col}={value}")
                row_text += ", ".join(row_items)
                full_text += row_text + "\n"

            # Add comprehensive numerical analysis
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                full_text += f"\n=== NUMERICAL ANALYSIS ===\n"
                for col in numerical_cols:
                    col_data = df[col].dropna()
                    if not col_data.empty:
                        sorted_values = col_data.sort_values(ascending=False)
                        full_text += f"\n{col} Statistics:\n"
                        full_text += f"  Maximum: {col_data.max()}\n"
                        full_text += f"  Minimum: {col_data.min()}\n"
                        full_text += f"  Mean: {col_data.mean():.2f}\n"
                        full_text += f"  Top 5 values: {sorted_values.head().tolist()}\n"

            documents.append(Document(page_content=full_text))
            return documents

        # For larger datasets, use overlapping chunks
        chunk_size = 25
        overlap = 5

        for i in range(0, len(df), chunk_size - overlap):
            chunk_df = df.iloc[i:i+chunk_size]

            chunk_text = f"CSV Data Chunk {i//(chunk_size-overlap) + 1} (Rows {i+1}-{min(i+chunk_size, len(df))}):\n"
            chunk_text += f"Columns: {', '.join(df.columns)}\n\n"

            for idx, row in chunk_df.iterrows():
                row_text = f"Row {idx + 1}: "
                row_items = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        row_items.append(f"{col}={value}")
                row_text += ", ".join(row_items)
                chunk_text += row_text + "\n"

            # Add numerical summaries
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                chunk_text += f"\nNumerical Summary for this chunk:\n"
                for col in numerical_cols:
                    chunk_data = chunk_df[col]
                    if not chunk_data.empty:
                        chunk_text += f"{col}: min={chunk_data.min()}, max={chunk_data.max()}, mean={chunk_data.mean():.2f}\n"

            documents.append(Document(page_content=chunk_text))

        return documents

    def _create_embeddings_with_retry(self, batch, embeddings, max_retries=3):
        """
        Create embeddings for a batch with retry logic and dimension validation
        """
        for attempt in range(max_retries):
            try:
                valid_batch = [doc for doc in batch if doc.page_content.strip()]
                if not valid_batch:
                    return None

                return FAISS.from_documents(valid_batch, embeddings)
            except Exception as e:
                error_msg = str(e).lower()

                if "permission" in error_msg or "scope" in error_msg or "rate" in error_msg:
                    st.warning(f"API rate limit hit, retrying in {wait_time:.1f}s...")

                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        st.info(f"API rate limit hit, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.warning(f"Max retries reached for batch after {max_retries} attempts")
                        return None
                else:
                    st.warning(f"Batch error: {str(e)}")
                    return None
        return None

    def _estimate_tokens(self, text):
        """Estimate token count for text (roughly 4 chars per token)"""
        return len(text) // 4

    def _create_embeddings_single_batch(self, split_docs, embeddings):
        """
        Create embeddings with token-aware batching to avoid API limits
        """
        try:
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"ENTRY: _create_embeddings_single_batch with {len(split_docs)} docs\n")

            valid_docs = [doc for doc in split_docs if doc.page_content.strip()]

            if not valid_docs:
                st.error("No valid documents to process")
                with open(f'{log_dir}/embedder.log', 'a') as f:
                    f.write(f"ERROR: No valid documents found\n")
                return None
        except Exception as e:
            error_msg = f"Error in _create_embeddings_single_batch entry: {str(e)}"
            with open(f'{log_dir}/embedder.log', 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            st.error(error_msg)
            return None

        # Token-aware batching
        max_tokens_per_batch = 250000  # Safe limit below 300k
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in valid_docs:
            doc_tokens = self._estimate_tokens(doc.page_content)

            if current_tokens + doc_tokens > max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        with open(f'{log_dir}/embedder.log', 'a') as f:
            f.write(f"STEP: Split into {len(batches)} token-aware batches\n")

        st.info(f"Processing {len(valid_docs)} documents in {len(batches)} batches")

        # Process batches sequentially
        all_vectors = None

        for i, batch in enumerate(batches):
            with st.spinner(f"Processing batch {i+1}/{len(batches)} ({len(batch)} documents)..."):
                try:
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"STEP: Processing batch {i+1} with {len(batch)} docs\n")

                    if all_vectors is None:
                        # First batch - create initial FAISS index
                        all_vectors = FAISS.from_documents(batch, embeddings)
                        with open(f'{log_dir}/embedder.log', 'a') as f:
                            f.write(f"SUCCESS: Created initial FAISS index with batch {i+1}\n")
                    else:
                        # Subsequent batches - create separate index and merge
                        batch_vectors = FAISS.from_documents(batch, embeddings)
                        all_vectors.merge_from(batch_vectors)
                        with open(f'{log_dir}/embedder.log', 'a') as f:
                            f.write(f"SUCCESS: Merged batch {i+1} into main index\n")

                    # Add small delay between batches to avoid rate limits
                    if i < len(batches) - 1:
                        time.sleep(1)

                except Exception as e:
                    error_msg = f"Batch {i+1} failed: {str(e)}"
                    with open(f'{log_dir}/embedder.log', 'a') as f:
                        f.write(f"ERROR: {error_msg}\n")

                    if "max_tokens_per_request" in str(e):
                        st.error(f"Batch {i+1} still too large. Skipping this batch.")
                        continue
                    else:
                        st.error(error_msg)
                        return None

        if all_vectors is None:
            st.error("Failed to create any embeddings")
            return None

        # Success - return the combined vectors
        with open(f'{log_dir}/embedder.log', 'a') as f:
            f.write(f"SUCCESS: Combined all batches into final FAISS index\n")

        st.success(f"Successfully created embeddings for {len(valid_docs)} documents!")
        return all_vectors

    def _process_large_csv(self, df):
        """
        Legacy method - kept for backward compatibility
        """

    def _debug_embedding_dimensions(self, documents, embeddings):
        """
        Debug method to validate embedding dimensions and FAISS compatibility for all documents
        """
        try:
            dimensions = []
            problematic_docs = []

            debug_info = []
            debug_info.append("=== EMBEDDING DIMENSION DEBUG ===")
            debug_info.append(f"Total documents to validate: {len(documents)}")

            # Test embedding creation for each document
            for i, doc in enumerate(documents):
                try:
                    content = doc.page_content[:500]
                    embedding = embeddings.embed_query(content)

                    # Log to file
                    with open(f'{log_dir}/debug_dimensions.log', 'a') as f:
                        f.write(f"Document {i}: content_length={len(content)}, embedding_length={len(embedding)}\n")

                    if not embedding:
                        error_msg = f"Document {i}: Empty embedding"
                        debug_info.append(error_msg)
                        logging.error(error_msg)
                        problematic_docs.append(i)
                        continue

                    dim = len(embedding)
                    dimensions.append(dim)

                    # Test numpy conversion
                    np_embedding = np.array(embedding).astype('float32')

                    if np_embedding.shape != (dim,):
                        error_msg = f"Document {i}: Invalid numpy shape {np_embedding.shape}"
                        debug_info.append(error_msg)
                        logging.error(error_msg)
                        problematic_docs.append(i)

                    # Log individual document results
                    debug_info.append(f"Document {i}: dimension={dim}, numpy_shape={np_embedding.shape}")

                except Exception as e:
                    error_msg = f"Document {i}: Embedding failed - {str(e)}"
                    debug_info.append(error_msg)
                    logging.error(error_msg)
                    problematic_docs.append(i)

            # Comprehensive analysis
            unique_dims = set(dimensions)
            debug_info.append(f"=== ANALYSIS RESULTS ===")
            debug_info.append(f"Found dimensions: {sorted(unique_dims)}")
            debug_info.append(f"Documents with issues: {len(problematic_docs)}")
            debug_info.append(f"Problematic document indices: {problematic_docs}")



            with open(f'{log_dir}/debug_dimensions.log', 'a') as f:
                f.write(f"Analysis complete:\n")
                f.write(f"Unique dimensions: {sorted(unique_dims)}\n")
                f.write(f"Problematic documents: {problematic_docs}\n")

            # Streamlit output
            for info in debug_info:
                logging.info(info)

            # Validate consistency
            if len(unique_dims) > 1:
                error_msg = f"Inconsistent embedding dimensions detected: {sorted(unique_dims)}"
                logging.error(error_msg)

                # Additional detailed logging
                with open(f'{log_dir}/debug_dimension_error.log', 'w') as f:
                    f.write(f"DIMENSION MISMATCH ERROR\n")
                    f.write(f"Expected: 1536\n")
                    f.write(f"Found: {sorted(unique_dims)}\n")
                    f.write(f"Documents: {len(documents)}\n")
                    for i, doc in enumerate(documents[:5]):  # Log first 5 documents
                        f.write(f"Doc {i}: content_length={len(doc.page_content)}\n")

                return False

            # Test FAISS compatibility
            if unique_dims:
                dim = list(unique_dims)[0]
                success_msg = f"All embeddings have consistent dimension: {dim}"
                logging.info(success_msg)

                # Test with FAISS
                try:
                    faiss = dependable_faiss_import()
                    test_vectors = np.random.rand(1, dim).astype('float32')
                    index = faiss.IndexFlatIP(dim)
                    index.add(test_vectors)

                    faiss_success = "FAISS compatibility test passed"
                    logging.info(faiss_success)

                    # Log FAISS test
                    with open(f'{log_dir}/faiss_test.log', 'w') as f:
                        f.write(f"FAISS test successful\n")
                        f.write(f"Dimension: {dim}\n")
                        f.write(f"Test vectors shape: {test_vectors.shape}\n")

                except Exception as e:
                    faiss_error = f"FAISS compatibility test failed: {str(e)}"
                    logging.error(faiss_error)

                    with open(f'{log_dir}/faiss_error.log', 'w') as f:
                        f.write(f"FAISS test failed: {str(e)}\n")

                    return False

            return True

        except Exception as e:
            error_msg = f"Debug validation failed: {str(e)}"
            logging.error(error_msg)

            with open(f'{log_dir}/debug_error.log', 'w') as f:
                f.write(f"Debug validation error: {str(e)}\n")

            return False

    def _create_embeddings_fallback(self, documents, embeddings):
        """
        Fallback method for creating embeddings when the main method fails
        """
        try:
            st.info("Using fallback embedding method...")

            # Process documents in smaller batches
            batch_size = 10
            all_batches = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                valid_batch = [doc for doc in batch if doc.page_content.strip()]

                if valid_batch:
                    try:
                        # Create embeddings for this batch
                        batch_vectors = FAISS.from_documents(valid_batch, embeddings)
                        all_batches.append(batch_vectors)
                        st.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    except Exception as e:
                        st.warning(f"Skipped batch {i//batch_size + 1} due to error: {str(e)}")
                        continue

            if not all_batches:
                st.error("All batches failed to process")
                return None

            # Merge all successful batches
            if len(all_batches) == 1:
                return all_batches[0]

            # Merge multiple batches
            merged_vectors = all_batches[0]
            for batch_vectors in all_batches[1:]:
                try:
                    merged_vectors.merge_from(batch_vectors)
                except Exception as e:
                    st.warning(f"Failed to merge batch: {str(e)}")
                    continue

            st.success(f"Successfully created embeddings using fallback method with {len(all_batches)} batches")
            return merged_vectors

        except Exception as e:
            st.error(f"Fallback method also failed: {str(e)}")
            return None
