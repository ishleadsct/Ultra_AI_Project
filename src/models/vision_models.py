"""
Ultra AI Project - Vision Models

Computer vision model implementations for image analysis, object detection,
OCR, and visual understanding capabilities.
"""

import asyncio
import base64
import io
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image
import numpy as np
from pydantic import BaseModel, Field, validator

from ..utils.logger import get_logger
from ..utils.file_handler import FileHandler
from . import ModelUsage, StandardResponse, create_success_response, create_error_response

logger = get_logger(__name__)

class VisionTask(Enum):
    """Vision task types."""
    IMAGE_ANALYSIS = "image_analysis"
    OBJECT_DETECTION = "object_detection"
    OCR = "ocr"
    IMAGE_CLASSIFICATION = "image_classification"
    FACE_DETECTION = "face_detection"
    IMAGE_DESCRIPTION = "image_description"
    VISUAL_QA = "visual_qa"
    IMAGE_SIMILARITY = "image_similarity"
    IMAGE_GENERATION = "image_generation"

class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"

@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float
    confidence: Optional[float] = None

@dataclass
class DetectedObject:
    """Detected object information."""
    label: str
    confidence: float
    bounding_box: BoundingBox
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OCRResult:
    """OCR result structure."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    language: Optional[str] = None

@dataclass
class FaceDetection:
    """Face detection result."""
    bounding_box: BoundingBox
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

class ImageAnalysisRequest(BaseModel):
    """Image analysis request structure."""
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image")
    image_path: Optional[str] = Field(None, description="Local file path to image")
    file_id: Optional[str] = Field(None, description="File ID from file handler")
    task: VisionTask = Field(VisionTask.IMAGE_ANALYSIS, description="Vision task to perform")
    prompt: Optional[str] = Field(None, description="Text prompt for the analysis")
    model: Optional[str] = Field(None, description="Specific model to use")
    max_tokens: Optional[int] = Field(300, description="Maximum tokens for description")
    detail_level: str = Field("auto", description="Detail level: low, high, auto")
    language: Optional[str] = Field("en", description="Language for OCR/text analysis")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold")
    return_coordinates: bool = Field(True, description="Return bounding box coordinates")
    
    @validator('image_data', 'image_url', 'image_path', 'file_id')
    def validate_image_source(cls, v, values):
        """Ensure at least one image source is provided."""
        sources = [values.get('image_data'), values.get('image_url'), 
                  values.get('image_path'), v]
        if not any(sources):
            raise ValueError("At least one image source must be provided")
        return v

class ImageAnalysisResponse(BaseModel):
    """Image analysis response structure."""
    task: VisionTask
    success: bool
    description: Optional[str] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    ocr_results: List[OCRResult] = Field(default_factory=list)
    faces: List[FaceDetection] = Field(default_factory=list)
    classifications: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model: str
    provider: str
    usage: Optional[ModelUsage] = None
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

class VisionModelManager:
    """Manager for computer vision models and tasks."""
    
    def __init__(self, model_manager=None, file_handler: Optional[FileHandler] = None):
        self.model_manager = model_manager
        self.file_handler = file_handler
        
        # Vision-specific configurations
        self.max_image_size = 20 * 1024 * 1024  # 20MB
        self.supported_formats = {fmt.value for fmt in ImageFormat}
        
        # Model capabilities mapping
        self.model_capabilities = {
            "openai_gpt4_vision": [
                VisionTask.IMAGE_ANALYSIS,
                VisionTask.IMAGE_DESCRIPTION,
                VisionTask.VISUAL_QA,
                VisionTask.OCR
            ],
            "openai_dalle3": [
                VisionTask.IMAGE_GENERATION
            ]
        }
        
        logger.info("VisionModelManager initialized")
    
    async def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResponse:
        """Analyze image based on specified task."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Load and validate image
            image_data = await self._load_image(request)
            if not image_data:
                return ImageAnalysisResponse(
                    task=request.task,
                    success=False,
                    error="Failed to load image",
                    model="unknown",
                    provider="unknown",
                    processing_time=0.0
                )
            
            # Select appropriate model
            model_config = await self._select_vision_model(request.task, request.model)
            if not model_config:
                return ImageAnalysisResponse(
                    task=request.task,
                    success=False,
                    error=f"No available models for task: {request.task.value}",
                    model="unknown",
                    provider="unknown",
                    processing_time=0.0
                )
            
            # Perform analysis based on task type
            if request.task == VisionTask.IMAGE_ANALYSIS:
                result = await self._analyze_image_general(image_data, request, model_config)
            elif request.task == VisionTask.IMAGE_DESCRIPTION:
                result = await self._describe_image(image_data, request, model_config)
            elif request.task == VisionTask.VISUAL_QA:
                result = await self._visual_question_answering(image_data, request, model_config)
            elif request.task == VisionTask.OCR:
                result = await self._perform_ocr(image_data, request, model_config)
            elif request.task == VisionTask.OBJECT_DETECTION:
                result = await self._detect_objects(image_data, request, model_config)
            elif request.task == VisionTask.FACE_DETECTION:
                result = await self._detect_faces(image_data, request, model_config)
            elif request.task == VisionTask.IMAGE_CLASSIFICATION:
                result = await self._classify_image(image_data, request, model_config)
            elif request.task == VisionTask.IMAGE_GENERATION:
                result = await self._generate_image(request, model_config)
            else:
                raise ValueError(f"Unsupported vision task: {request.task.value}")
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Image analysis failed: {e}")
            
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=str(e),
                model="unknown",
                provider="unknown",
                processing_time=processing_time
            )
    
    async def _load_image(self, request: ImageAnalysisRequest) -> Optional[bytes]:
        """Load image data from various sources."""
        try:
            # From base64 data
            if request.image_data:
                return base64.b64decode(request.image_data)
            
            # From URL
            if request.image_url:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(request.image_url)
                    response.raise_for_status()
                    return response.content
            
            # From local file path
            if request.image_path:
                with open(request.image_path, 'rb') as f:
                    return f.read()
            
            # From file handler
            if request.file_id and self.file_handler:
                return await self.file_handler.read_file(request.file_id, mode='rb')
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    async def _select_vision_model(self, task: VisionTask, preferred_model: Optional[str] = None) -> Optional[Any]:
        """Select appropriate vision model for task."""
        if not self.model_manager:
            logger.error("Model manager not available")
            return None
        
        # Use preferred model if specified and capable
        if preferred_model:
            model_config = self.model_manager.models.get(preferred_model)
            if (model_config and 
                preferred_model in self.model_capabilities and
                task in self.model_capabilities[preferred_model]):
                return model_config
        
        # Find capable models
        capable_models = []
        for model_name, capabilities in self.model_capabilities.items():
            if task in capabilities and model_name in self.model_manager.models:
                capable_models.append(self.model_manager.models[model_name])
        
        if not capable_models:
            return None
        
        # Select best model using model manager's selection logic
        return self.model_manager.router.select_model(
            capable_models, 
            self.model_manager.model_metrics
        )
    
    async def _analyze_image_general(self, image_data: bytes, request: ImageAnalysisRequest, 
                                   model_config: Any) -> ImageAnalysisResponse:
        """Perform general image analysis."""
        try:
            if model_config.provider == "openai":
                return await self._openai_vision_analysis(image_data, request, model_config)
            else:
                raise ValueError(f"Provider not supported for vision: {model_config.provider}")
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"General analysis failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _describe_image(self, image_data: bytes, request: ImageAnalysisRequest,
                            model_config: Any) -> ImageAnalysisResponse:
        """Generate image description."""
        try:
            if model_config.provider == "openai":
                # Use specific prompt for description
                description_request = ImageAnalysisRequest(
                    **request.dict(),
                    prompt="Describe this image in detail, including objects, people, setting, colors, and mood."
                )
                return await self._openai_vision_analysis(image_data, description_request, model_config)
            else:
                raise ValueError(f"Provider not supported for description: {model_config.provider}")
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"Image description failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _visual_question_answering(self, image_data: bytes, request: ImageAnalysisRequest,
                                       model_config: Any) -> ImageAnalysisResponse:
        """Answer questions about the image."""
        try:
            if not request.prompt:
                raise ValueError("Prompt required for visual question answering")
            
            if model_config.provider == "openai":
                return await self._openai_vision_analysis(image_data, request, model_config)
            else:
                raise ValueError(f"Provider not supported for VQA: {model_config.provider}")
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"Visual QA failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _perform_ocr(self, image_data: bytes, request: ImageAnalysisRequest,
                          model_config: Any) -> ImageAnalysisResponse:
        """Perform OCR on the image."""
        try:
            if model_config.provider == "openai":
                # Use specific prompt for OCR
                ocr_request = ImageAnalysisRequest(
                    **request.dict(),
                    prompt="Extract all text from this image. Provide the text content exactly as it appears."
                )
                result = await self._openai_vision_analysis(image_data, ocr_request, model_config)
                
                # Convert description to OCR results
                if result.success and result.description:
                    result.ocr_results = [
                        OCRResult(
                            text=result.description,
                            confidence=0.8,  # Estimated confidence
                            language=request.language
                        )
                    ]
                
                return result
            else:
                # Try using local OCR libraries
                return await self._local_ocr(image_data, request, model_config)
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"OCR failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _detect_objects(self, image_data: bytes, request: ImageAnalysisRequest,
                            model_config: Any) -> ImageAnalysisResponse:
        """Detect objects in the image."""
        try:
            if model_config.provider == "openai":
                # Use specific prompt for object detection
                detection_request = ImageAnalysisRequest(
                    **request.dict(),
                    prompt="Identify and list all objects visible in this image with their approximate locations."
                )
                return await self._openai_vision_analysis(image_data, detection_request, model_config)
            else:
                # Try using local object detection models
                return await self._local_object_detection(image_data, request, model_config)
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"Object detection failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _detect_faces(self, image_data: bytes, request: ImageAnalysisRequest,
                          model_config: Any) -> ImageAnalysisResponse:
        """Detect faces in the image."""
        try:
            # Use local face detection (OpenCV, face_recognition, etc.)
            return await self._local_face_detection(image_data, request, model_config)
            
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"Face detection failed: {str(e)}",
                model=model_config.name if model_config else "local",
                provider=model_config.provider if model_config else "local",
                processing_time=0.0
            )
    
    async def _classify_image(self, image_data: bytes, request: ImageAnalysisRequest,
                            model_config: Any) -> ImageAnalysisResponse:
        """Classify the image."""
        try:
            if model_config.provider == "openai":
                # Use specific prompt for classification
                classification_request = ImageAnalysisRequest(
                    **request.dict(),
                    prompt="Classify this image into categories. What type of scene, object, or subject is this?"
                )
                return await self._openai_vision_analysis(image_data, classification_request, model_config)
            else:
                # Try using local classification models
                return await self._local_classification(image_data, request, model_config)
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"Image classification failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _generate_image(self, request: ImageAnalysisRequest, model_config: Any) -> ImageAnalysisResponse:
        """Generate image from text prompt."""
        try:
            if not request.prompt:
                raise ValueError("Prompt required for image generation")
            
            if model_config.provider == "openai":
                return await self._openai_image_generation(request, model_config)
            else:
                raise ValueError(f"Provider not supported for generation: {model_config.provider}")
                
        except Exception as e:
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error=f"Image generation failed: {str(e)}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0
            )
    
    async def _openai_vision_analysis(self, image_data: bytes, request: ImageAnalysisRequest,
                                    model_config: Any) -> ImageAnalysisResponse:
        """Perform vision analysis using OpenAI GPT-4 Vision."""
        try:
            # Get model instance
            model_instance = self.model_manager.model_instances.get(model_config.name)
            if not model_instance:
                raise ValueError(f"Model instance not available: {model_config.name}")
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request.prompt or "Analyze this image and describe what you see."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": request.detail_level
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = await model_instance.chat.completions.create(
                model=model_config.model_id or "gpt-4-vision-preview",
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=0.1
            )
            
            # Extract response
            description = response.choices[0].message.content if response.choices else ""
            
            # Create usage info
            usage = None
            if response.usage:
                usage = ModelUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
                
                if model_config.cost_per_1k_tokens:
                    input_cost = (usage.prompt_tokens / 1000) * model_config.cost_per_1k_tokens.get("input", 0)
                    output_cost = (usage.completion_tokens / 1000) * model_config.cost_per_1k_tokens.get("output", 0)
                    usage.estimated_cost = input_cost + output_cost
            
            return ImageAnalysisResponse(
                task=request.task,
                success=True,
                description=description,
                model=model_config.name,
                provider=model_config.provider,
                usage=usage,
                processing_time=0.0,  # Will be set by caller
                metadata={
                    "image_size": len(image_data),
                    "detail_level": request.detail_level
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI vision analysis failed: {e}")
            raise
    
    async def _openai_image_generation(self, request: ImageAnalysisRequest, 
                                     model_config: Any) -> ImageAnalysisResponse:
        """Generate image using OpenAI DALL-E."""
        try:
            # Get model instance
            model_instance = self.model_manager.model_instances.get(model_config.name)
            if not model_instance:
                raise ValueError(f"Model instance not available: {model_config.name}")
            
            # Generate image
            response = await model_instance.images.generate(
                model=model_config.model_id or "dall-e-3",
                prompt=request.prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            # Get generated image URL
            image_url = response.data[0].url if response.data else None
            if not image_url:
                raise ValueError("No image generated")
            
            return ImageAnalysisResponse(
                task=request.task,
                success=True,
                description=f"Generated image from prompt: {request.prompt}",
                model=model_config.name,
                provider=model_config.provider,
                processing_time=0.0,  # Will be set by caller
                metadata={
                    "generated_image_url": image_url,
                    "prompt": request.prompt
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {e}")
            raise
    
    async def _local_ocr(self, image_data: bytes, request: ImageAnalysisRequest,
                        model_config: Any) -> ImageAnalysisResponse:
        """Perform OCR using local libraries."""
        try:
            # This would use libraries like pytesseract, easyocr, etc.
            # For now, return a placeholder implementation
            
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error="Local OCR not yet implemented",
                model="local_ocr",
                provider="local",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Local OCR failed: {e}")
            raise
    
    async def _local_object_detection(self, image_data: bytes, request: ImageAnalysisRequest,
                                    model_config: Any) -> ImageAnalysisResponse:
        """Perform object detection using local models."""
        try:
            # This would use libraries like YOLO, OpenCV, etc.
            # For now, return a placeholder implementation
            
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error="Local object detection not yet implemented",
                model="local_yolo",
                provider="local",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Local object detection failed: {e}")
            raise
    
    async def _local_face_detection(self, image_data: bytes, request: ImageAnalysisRequest,
                                  model_config: Any) -> ImageAnalysisResponse:
        """Perform face detection using local libraries."""
        try:
            # This would use libraries like face_recognition, OpenCV, etc.
            # For now, return a placeholder implementation
            
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error="Local face detection not yet implemented",
                model="local_face",
                provider="local",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Local face detection failed: {e}")
            raise
    
    async def _local_classification(self, image_data: bytes, request: ImageAnalysisRequest,
                                  model_config: Any) -> ImageAnalysisResponse:
        """Perform image classification using local models."""
        try:
            # This would use libraries like transformers, torchvision, etc.
            # For now, return a placeholder implementation
            
            return ImageAnalysisResponse(
                task=request.task,
                success=False,
                error="Local classification not yet implemented",
                model="local_classifier",
                provider="local",
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Local classification failed: {e}")
            raise
    
    def get_supported_tasks(self) -> List[VisionTask]:
        """Get list of supported vision tasks."""
        return list(VisionTask)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return list(self.supported_formats)
    
    def validate_image(self, image_data: bytes) -> Tuple[bool, Optional[str]]:
        """Validate image data."""
        try:
            # Check size
            if len(image_data) > self.max_image_size:
                return False, f"Image too large: {len(image_data)} bytes > {self.max_image_size} bytes"
            
            # Check format using PIL
            image = Image.open(io.BytesIO(image_data))
            format_lower = image.format.lower() if image.format else "unknown"
            
            if format_lower not in self.supported_formats:
                return False, f"Unsupported format: {format_lower}"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"
    
    async def batch_analyze(self, requests: List[ImageAnalysisRequest]) -> List[ImageAnalysisResponse]:
        """Analyze multiple images in batch."""
        tasks = [self.analyze_image(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_model_capabilities(self) -> Dict[str, List[str]]:
        """Get model capabilities mapping."""
        return {
            model: [task.value for task in tasks]
            for model, tasks in self.model_capabilities.items()
        }
