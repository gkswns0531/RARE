import os
import pdfplumber
import tiktoken
from pathlib import Path
from typing import List, Dict, Any

# Compatible import: prefer lightweight text-splitters package
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore


class DocumentProcessor:
    """Document processor for RARE - handles PDF to text conversion and chunking"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0, chunking_method: str = "sentence"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method  # "sentence" or "token"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configure RecursiveCharacterTextSplitter for academic document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.tokenizer.encode(text)),
            separators=self._get_korean_aware_separators(),
            keep_separator=True,
            strip_whitespace=True,
            add_start_index=False
        )
    
    def _get_korean_aware_separators(self) -> List[str]:
        """
        Define hierarchical separators optimized for Korean academic documents.
        
        Returns:
            List[str]: Separators in order of preference for text splitting
        """
        return [
            # Document structure separators
            "\n\n\n",  # Section breaks
            "\n\n",    # Paragraph breaks
        
            # List and enumeration patterns
            "\n• ",    # Bullet points
            "\n- ",    # Dash points
            "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ",  # Common numbered lists
            "\n1 ", "\n2 ", "\n3 ", "\n4 ", "\n5 ",  # Common numbered items
            
            # General punctuation
            ".\n", ". ", ".\t",
            "!\n", "! ", "!\t",
            "?\n", "? ", "?\t",
            
            # Final fallback separators
            "\n", "\t", " ", ""
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF file, page by page"""
        page_texts = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        page_texts[page_num] = text.strip()
                        
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
        
        return page_texts
    
    def chunk_text(self, text: str, file_name: str, page_no: int) -> List[Dict[str, Any]]:
        """
        Chunk text using Korean-aware RecursiveCharacterTextSplitter.
        
        Args:
            text: Input text to be chunked
            file_name: Name of the source file
            page_no: Page number in the source document
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text.strip():
            return []
        
        text_chunks = self.text_splitter.split_text(text)
        
        chunks = []
        for idx, chunk_text in enumerate(text_chunks, 1):
            chunk_tokens = len(self.tokenizer.encode(chunk_text))
            
            chunk = {
                "file_name": file_name,
                "page_no": page_no,
                "content": chunk_text.strip(),
                "sub_text_index": str(idx),
                "token_count": chunk_tokens
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_pdf_to_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF file to text chunks"""
        
        file_name = Path(pdf_path).name
        
        # Extract text from PDF
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        if not page_texts:
            raise Exception(f"No text extracted from PDF: {pdf_path}")
        
        # Generate chunks from all pages
        all_chunks = []
        
        for page_no, text in page_texts.items():
            page_chunks = self.chunk_text(text, file_name, page_no)
            all_chunks.extend(page_chunks)
        
        print(f"Processed {file_name}: {len(page_texts)} pages → {len(all_chunks)} chunks")
        return all_chunks
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF files to text chunks"""
        
        all_chunks = []
        
        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf_to_chunks(pdf_path)
                all_chunks.extend(chunks)
                print(f"Processed: {Path(pdf_path).name}")
            except Exception as e:
                print(f"Error: Failed to process {Path(pdf_path).name}: {e}")
        
        print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_paths)} PDF files")
        return all_chunks
    
    def process_pdf_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a folder"""
        
        folder = Path(folder_path)
        if not folder.exists():
            raise Exception(f"Folder not found: {folder_path}")
        
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            raise Exception(f"No PDF files found in: {folder_path}")
        
        print(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        pdf_paths = [str(pdf) for pdf in pdf_files]
        return self.process_multiple_pdfs(pdf_paths)
