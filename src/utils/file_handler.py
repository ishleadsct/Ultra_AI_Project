"""
Ultra AI Project - File Handler

Comprehensive file management utilities for handling uploads, downloads,
processing, and metadata extraction across various file types.
"""

import os
import hashlib
import mimetypes
import magic
import tempfile
import shutil
import asyncio
import aiofiles
from typing import Dict, List, Optional, Any, Tuple, Union, IO
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import json
import zipfile
import tarfile
from PIL import Image
import fitz  # PyMuPDF for PDF processing

from .logger import get_logger
from .helpers import sanitize_string, format_bytes, generate_file_id

logger = get_logger(__name__)

@dataclass
class FileMetadata:
    """File metadata information."""
    id: str
    filename: str
    original_filename: str
    file_path: str
    mime_type: str
    file_size: int
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    checksum: str
    encoding: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "file_size_formatted": format_bytes(self.file_size),
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "checksum": self.checksum,
            "encoding": self.encoding,
            "dimensions": self.dimensions,
            "duration": self.duration,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "tags": self.tags
        }

class FileTypeDetector:
    """File type detection and validation."""
    
    def __init__(self):
        self.magic_mime = magic.Magic(mime=True)
        self.magic_type = magic.Magic()
    
    def detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type using python-magic."""
        try:
            return self.magic_mime.from_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to detect MIME type with magic: {e}")
            # Fallback to mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type or "application/octet-stream"
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect human-readable file type."""
        try:
            return self.magic_type.from_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to detect file type: {e}")
            return "Unknown file type"
    
    def is_text_file(self, file_path: str) -> bool:
        """Check if file is a text file."""
        mime_type = self.detect_mime_type(file_path)
        return mime_type.startswith('text/') or mime_type in [
            'application/json',
            'application/yaml',
            'application/xml',
            'application/javascript'
        ]
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if file is an image."""
        mime_type = self.detect_mime_type(file_path)
        return mime_type.startswith('image/')
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if file is audio."""
        mime_type = self.detect_mime_type(file_path)
        return mime_type.startswith('audio/')
    
    def is_video_file(self, file_path: str) -> bool:
        """Check if file is video."""
        mime_type = self.detect_mime_type(file_path)
        return mime_type.startswith('video/')
    
    def is_document_file(self, file_path: str) -> bool:
        """Check if file is a document."""
        mime_type = self.detect_mime_type(file_path)
        document_types = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        ]
        return mime_type in document_types

# Global file type detector instance
file_type_detector = FileTypeDetector()

class FileHandler:
    """Comprehensive file handling and management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Configuration
        self.storage_path = Path(self.config.get("storage_path", "runtime/storage"))
        self.temp_path = Path(self.config.get("temp_path", "runtime/temp"))
        self.upload_path = Path(self.config.get("upload_path", "runtime/uploads"))
        self.max_file_size = self.config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self.allowed_extensions = set(self.config.get("allowed_extensions", []))
        self.chunk_size = self.config.get("chunk_size", 8192)
        
        # File storage
        self.files: Dict[str, FileMetadata] = {}
        
        # Ensure directories exist
        for path in [self.storage_path, self.temp_path, self.upload_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileHandler initialized with storage at {self.storage_path}")
    
    async def upload_file(self, file_data: Union[bytes, IO], 
                         original_filename: str,
                         user_id: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> str:
        """Upload and store a file."""
        try:
            # Validate filename
            safe_filename = sanitize_string(original_filename)
            if not safe_filename:
                raise ValueError("Invalid filename")
            
            # Check file extension if restrictions are set
            if self.allowed_extensions:
                file_ext = Path(safe_filename).suffix.lower()
                if file_ext not in self.allowed_extensions:
                    raise ValueError(f"File type {file_ext} not allowed")
            
            # Generate unique file ID and path
            file_id = generate_file_id()
            file_ext = Path(safe_filename).suffix
            stored_filename = f"{file_id}{file_ext}"
            file_path = self.storage_path / stored_filename
            
            # Write file data
            if isinstance(file_data, bytes):
                # Check file size
                if len(file_data) > self.max_file_size:
                    raise ValueError(f"File size {len(file_data)} exceeds limit {self.max_file_size}")
                
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(file_data)
            else:
                # Stream file data
                total_size = 0
                async with aiofiles.open(file_path, 'wb') as f:
                    while True:
                        chunk = file_data.read(self.chunk_size)
                        if not chunk:
                            break
                        
                        total_size += len(chunk)
                        if total_size > self.max_file_size:
                            # Clean up partial file
                            await self._cleanup_file(file_path)
                            raise ValueError(f"File size exceeds limit {self.max_file_size}")
                        
                        await f.write(chunk)
            
            # Extract metadata
            metadata = await self._extract_metadata(file_path, safe_filename, original_filename)
            metadata.id = file_id
            
            # Add user metadata
            if user_id:
                metadata.metadata["user_id"] = user_id
            
            if tags:
                metadata.tags = tags
            
            # Store metadata
            self.files[file_id] = metadata
            
            # Save metadata to disk
            await self._save_metadata(metadata)
            
            logger.info(f"File uploaded: {original_filename} -> {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
    
    async def _extract_metadata(self, file_path: Path, filename: str, 
                               original_filename: str) -> FileMetadata:
        """Extract comprehensive file metadata."""
        try:
            stat = file_path.stat()
            
            # Basic metadata
            metadata = FileMetadata(
                id="",  # Will be set by caller
                filename=filename,
                original_filename=original_filename,
                file_path=str(file_path),
                mime_type=file_type_detector.detect_mime_type(str(file_path)),
                file_size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                accessed_at=datetime.fromtimestamp(stat.st_atime),
                checksum=await self._calculate_checksum(file_path)
            )
            
            # Type-specific metadata extraction
            await self._extract_type_specific_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed for {file_path}: {e}")
            raise
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(self.chunk_size)
                if not chunk:
                    break
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _extract_type_specific_metadata(self, metadata: FileMetadata):
        """Extract metadata specific to file type."""
        file_path = Path(metadata.file_path)
        
        try:
            if file_type_detector.is_image_file(str(file_path)):
                await self._extract_image_metadata(metadata)
            elif file_type_detector.is_audio_file(str(file_path)):
                await self._extract_audio_metadata(metadata)
            elif file_type_detector.is_video_file(str(file_path)):
                await self._extract_video_metadata(metadata)
            elif file_type_detector.is_document_file(str(file_path)):
                await self._extract_document_metadata(metadata)
            elif file_type_detector.is_text_file(str(file_path)):
                await self._extract_text_metadata(metadata)
                
        except Exception as e:
            logger.warning(f"Type-specific metadata extraction failed: {e}")
    
    async def _extract_image_metadata(self, metadata: FileMetadata):
        """Extract image-specific metadata."""
        try:
            with Image.open(metadata.file_path) as img:
                metadata.dimensions = img.size
                metadata.metadata.update({
                    "format": img.format,
                    "mode": img.mode,
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                })
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    metadata.metadata["exif"] = dict(img._getexif())
                    
        except Exception as e:
            logger.warning(f"Image metadata extraction failed: {e}")
    
    async def _extract_audio_metadata(self, metadata: FileMetadata):
        """Extract audio-specific metadata."""
        try:
            # This would use a library like mutagen or eyed3
            # For now, we'll just set basic info
            metadata.metadata["type"] = "audio"
            
        except Exception as e:
            logger.warning(f"Audio metadata extraction failed: {e}")
    
    async def _extract_video_metadata(self, metadata: FileMetadata):
        """Extract video-specific metadata."""
        try:
            # This would use a library like ffprobe or opencv
            # For now, we'll just set basic info
            metadata.metadata["type"] = "video"
            
        except Exception as e:
            logger.warning(f"Video metadata extraction failed: {e}")
    
    async def _extract_document_metadata(self, metadata: FileMetadata):
        """Extract document-specific metadata."""
        try:
            if metadata.mime_type == "application/pdf":
                await self._extract_pdf_metadata(metadata)
            else:
                metadata.metadata["type"] = "document"
                
        except Exception as e:
            logger.warning(f"Document metadata extraction failed: {e}")
    
    async def _extract_pdf_metadata(self, metadata: FileMetadata):
        """Extract PDF-specific metadata."""
        try:
            doc = fitz.open(metadata.file_path)
            
            metadata.page_count = doc.page_count
            metadata.metadata.update({
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "encrypted": doc.is_encrypted,
                "page_count": doc.page_count
            })
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PDF metadata extraction failed: {e}")
    
    async def _extract_text_metadata(self, metadata: FileMetadata):
        """Extract text file metadata."""
        try:
            async with aiofiles.open(metadata.file_path, 'r', encoding='utf-8') as f:
                content = await f.read(1000)  # Read first 1KB to detect encoding
                
            metadata.encoding = "utf-8"
            metadata.metadata.update({
                "type": "text",
                "sample_content": content[:200] + "..." if len(content) > 200 else content
            })
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    async with aiofiles.open(metadata.file_path, 'r', encoding=encoding) as f:
                        content = await f.read(1000)
                    metadata.encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"Text metadata extraction failed: {e}")
    
    async def get_file(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata by ID."""
        return self.files.get(file_id)
    
    async def read_file(self, file_id: str, mode: str = 'rb') -> Optional[bytes]:
        """Read file content."""
        metadata = self.files.get(file_id)
        if not metadata:
            return None
        
        try:
            file_path = Path(metadata.file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if mode == 'rb':
                async with aiofiles.open(file_path, 'rb') as f:
                    return await f.read()
            else:
                encoding = metadata.encoding or 'utf-8'
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                    return content.encode(encoding)
                    
        except Exception as e:
            logger.error(f"Failed to read file {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete a file and its metadata."""
        try:
            metadata = self.files.get(file_id)
            if not metadata:
                return False
            
            # Remove file
            file_path = Path(metadata.file_path)
            if file_path.exists():
                await self._cleanup_file(file_path)
            
            # Remove metadata file
            metadata_path = self._get_metadata_path(file_id)
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from memory
            del self.files[file_id]
            
            logger.info(f"File deleted: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def _cleanup_file(self, file_path: Path):
        """Clean up a file safely."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
    
    def _get_metadata_path(self, file_id: str) -> Path:
        """Get metadata file path."""
        return self.storage_path / f"{file_id}.metadata.json"
    
    async def _save_metadata(self, metadata: FileMetadata):
        """Save metadata to disk."""
        try:
            metadata_path = self._get_metadata_path(metadata.id)
            
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata.to_dict(), indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.id}: {e}")
    
    async def list_files(self, user_id: Optional[str] = None, 
                        tags: Optional[List[str]] = None,
                        mime_type: Optional[str] = None) -> List[FileMetadata]:
        """List files with optional filtering."""
        files = list(self.files.values())
        
        # Filter by user
        if user_id:
            files = [f for f in files if f.metadata.get("user_id") == user_id]
        
        # Filter by tags
        if tags:
            files = [f for f in files if any(tag in f.tags for tag in tags)]
        
        # Filter by MIME type
        if mime_type:
            files = [f for f in files if f.mime_type == mime_type]
        
        # Sort by creation date (newest first)
        files.sort(key=lambda x: x.created_at, reverse=True)
        
        return files
    
    async def create_archive(self, file_ids: List[str], 
                           archive_name: str = "archive") -> Optional[str]:
        """Create a ZIP archive from multiple files."""
        try:
            archive_id = generate_file_id()
            archive_filename = f"{archive_name}_{archive_id}.zip"
            archive_path = self.temp_path / archive_filename
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_id in file_ids:
                    metadata = self.files.get(file_id)
                    if metadata:
                        file_path = Path(metadata.file_path)
                        if file_path.exists():
                            zipf.write(file_path, metadata.original_filename)
            
            # Upload the archive as a new file
            with open(archive_path, 'rb') as f:
                archive_file_id = await self.upload_file(
                    f, archive_filename, tags=["archive"]
                )
            
            # Clean up temporary archive
            archive_path.unlink()
            
            return archive_file_id
            
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            return None
    
    async def get_file_stats(self) -> Dict[str, Any]:
        """Get file storage statistics."""
        total_files = len(self.files)
        total_size = sum(f.file_size for f in self.files.values())
        
        # Group by MIME type
        mime_types = {}
        for file_meta in self.files.values():
            mime_type = file_meta.mime_type
            if mime_type not in mime_types:
                mime_types[mime_type] = {"count": 0, "size": 0}
            mime_types[mime_type]["count"] += 1
            mime_types[mime_type]["size"] += file_meta.file_size
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "total_size_formatted": format_bytes(total_size),
            "mime_types": mime_types,
            "storage_path": str(self.storage_path),
            "max_file_size": self.max_file_size,
            "max_file_size_formatted": format_bytes(self.max_file_size)
        }
    
    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            for file_path in self.temp_path.glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary files")
                
        except Exception as e:
            logger.error(f"Temporary file cleanup failed: {e}")
