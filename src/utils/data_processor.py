"""
Ultra AI Project - Data Processor

Comprehensive data processing utilities for validation, transformation,
analysis, and manipulation of various data types and formats.
"""

import json
import csv
import yaml
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
import hashlib
import base64
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import io
import gzip
import zipfile

from .logger import get_logger
from .helpers import safe_json_loads, sanitize_string

logger = get_logger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataTransformationError(Exception):
    """Custom exception for data transformation errors."""
    pass

@dataclass
class DataSchema:
    """Data schema definition for validation."""
    fields: Dict[str, Dict[str, Any]]
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Result of data processing operation."""
    success: bool
    data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

class DataProcessor:
    """Comprehensive data processing and transformation utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_data_size = self.config.get("max_data_size", 100 * 1024 * 1024)  # 100MB
        self.chunk_size = self.config.get("chunk_size", 10000)
        self.timeout = self.config.get("timeout", 300)  # 5 minutes
        
        # Processing statistics
        self.stats = {
            "operations_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0,
            "data_processed_mb": 0.0
        }
        
        logger.info("DataProcessor initialized")
    
    async def validate_data(self, data: Any, schema: DataSchema) -> ProcessingResult:
        """Validate data against a schema."""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            if not isinstance(data, dict):
                errors.append("Data must be a dictionary")
                return ProcessingResult(False, errors=errors)
            
            # Check required fields
            for field in schema.required_fields:
                if field not in data:
                    errors.append(f"Required field '{field}' is missing")
            
            # Validate field types and constraints
            for field_name, field_config in schema.fields.items():
                if field_name in data:
                    field_value = data[field_name]
                    field_type = field_config.get("type")
                    
                    # Type validation
                    if field_type and not self._validate_field_type(field_value, field_type):
                        errors.append(f"Field '{field_name}' has invalid type")
                    
                    # Constraint validation
                    field_constraints = field_config.get("constraints", {})
                    constraint_errors = self._validate_constraints(field_name, field_value, field_constraints)
                    errors.extend(constraint_errors)
            
            # Check for unexpected fields
            unexpected_fields = set(data.keys()) - set(schema.fields.keys())
            if unexpected_fields:
                warnings.append(f"Unexpected fields: {', '.join(unexpected_fields)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=len(errors) == 0, processing_time=processing_time)
            
            return ProcessingResult(
                success=len(errors) == 0,
                data=data,
                errors=errors,
                warnings=warnings,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "list": list,
            "dict": dict,
            "datetime": (str, datetime)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it
    
    def _validate_constraints(self, field_name: str, value: Any, constraints: Dict[str, Any]) -> List[str]:
        """Validate field constraints."""
        errors = []
        
        # Min/Max length for strings and lists
        if "min_length" in constraints:
            min_length = constraints["min_length"]
            if hasattr(value, "__len__") and len(value) < min_length:
                errors.append(f"Field '{field_name}' length is below minimum {min_length}")
        
        if "max_length" in constraints:
            max_length = constraints["max_length"]
            if hasattr(value, "__len__") and len(value) > max_length:
                errors.append(f"Field '{field_name}' length exceeds maximum {max_length}")
        
        # Min/Max value for numbers
        if "min_value" in constraints:
            min_value = constraints["min_value"]
            if isinstance(value, (int, float)) and value < min_value:
                errors.append(f"Field '{field_name}' value is below minimum {min_value}")
        
        if "max_value" in constraints:
            max_value = constraints["max_value"]
            if isinstance(value, (int, float)) and value > max_value:
                errors.append(f"Field '{field_name}' value exceeds maximum {max_value}")
        
        # Pattern matching for strings
        if "pattern" in constraints and isinstance(value, str):
            pattern = constraints["pattern"]
            if not re.match(pattern, value):
                errors.append(f"Field '{field_name}' does not match required pattern")
        
        # Allowed values
        if "allowed_values" in constraints:
            allowed_values = constraints["allowed_values"]
            if value not in allowed_values:
                errors.append(f"Field '{field_name}' value not in allowed values")
        
        return errors
    
    async def parse_json(self, data: Union[str, bytes]) -> ProcessingResult:
        """Parse JSON data with error handling."""
        start_time = datetime.now()
        
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            parsed_data = safe_json_loads(data)
            if parsed_data is None:
                return ProcessingResult(False, errors=["Invalid JSON format"])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=True, processing_time=processing_time, data_size=len(data))
            
            return ProcessingResult(
                success=True,
                data=parsed_data,
                processing_time=processing_time,
                metadata={"format": "json", "size": len(data)}
            )
            
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    async def parse_csv(self, data: Union[str, bytes], 
                       delimiter: str = ",", 
                       has_header: bool = True) -> ProcessingResult:
        """Parse CSV data."""
        start_time = datetime.now()
        
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            # Use StringIO for CSV parsing
            csv_file = io.StringIO(data)
            
            if has_header:
                reader = csv.DictReader(csv_file, delimiter=delimiter)
                parsed_data = [row for row in reader]
                headers = reader.fieldnames
            else:
                reader = csv.reader(csv_file, delimiter=delimiter)
                parsed_data = [row for row in reader]
                headers = None
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=True, processing_time=processing_time, data_size=len(data))
            
            return ProcessingResult(
                success=True,
                data=parsed_data,
                processing_time=processing_time,
                metadata={
                    "format": "csv",
                    "size": len(data),
                    "rows": len(parsed_data),
                    "headers": headers,
                    "delimiter": delimiter
                }
            )
            
        except Exception as e:
            logger.error(f"CSV parsing failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    async def parse_yaml(self, data: Union[str, bytes]) -> ProcessingResult:
        """Parse YAML data."""
        start_time = datetime.now()
        
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            parsed_data = yaml.safe_load(data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=True, processing_time=processing_time, data_size=len(data))
            
            return ProcessingResult(
                success=True,
                data=parsed_data,
                processing_time=processing_time,
                metadata={"format": "yaml", "size": len(data)}
            )
            
        except Exception as e:
            logger.error(f"YAML parsing failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    async def parse_xml(self, data: Union[str, bytes]) -> ProcessingResult:
        """Parse XML data."""
        start_time = datetime.now()
        
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            root = ET.fromstring(data)
            parsed_data = self._xml_to_dict(root)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=True, processing_time=processing_time, data_size=len(data))
            
            return ProcessingResult(
                success=True,
                data=parsed_data,
                processing_time=processing_time,
                metadata={"format": "xml", "size": len(data)}
            )
            
        except Exception as e:
            logger.error(f"XML parsing failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result["@attributes"] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:
                return element.text.strip()
            result["#text"] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            
            if child.tag in result:
                # Multiple elements with same tag - convert to list
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    async def transform_data(self, data: Any, transformations: List[Dict[str, Any]]) -> ProcessingResult:
        """Apply a series of transformations to data."""
        start_time = datetime.now()
        errors = []
        warnings = []
        current_data = data
        
        try:
            for i, transformation in enumerate(transformations):
                transform_type = transformation.get("type")
                transform_config = transformation.get("config", {})
                
                try:
                    if transform_type == "filter":
                        current_data = await self._apply_filter(current_data, transform_config)
                    elif transform_type == "map":
                        current_data = await self._apply_map(current_data, transform_config)
                    elif transform_type == "aggregate":
                        current_data = await self._apply_aggregation(current_data, transform_config)
                    elif transform_type == "sort":
                        current_data = await self._apply_sort(current_data, transform_config)
                    elif transform_type == "group":
                        current_data = await self._apply_grouping(current_data, transform_config)
                    elif transform_type == "clean":
                        current_data = await self._apply_cleaning(current_data, transform_config)
                    else:
                        warnings.append(f"Unknown transformation type: {transform_type}")
                        
                except Exception as e:
                    errors.append(f"Transformation {i+1} failed: {str(e)}")
                    break
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=len(errors) == 0, processing_time=processing_time)
            
            return ProcessingResult(
                success=len(errors) == 0,
                data=current_data,
                errors=errors,
                warnings=warnings,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    async def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filter transformation."""
        if not isinstance(data, list):
            raise DataTransformationError("Filter requires list data")
        
        field = config.get("field")
        operator = config.get("operator", "eq")
        value = config.get("value")
        
        if not field:
            raise DataTransformationError("Filter requires field specification")
        
        filtered_data = []
        for item in data:
            if isinstance(item, dict) and field in item:
                item_value = item[field]
                
                if operator == "eq" and item_value == value:
                    filtered_data.append(item)
                elif operator == "ne" and item_value != value:
                    filtered_data.append(item)
                elif operator == "gt" and item_value > value:
                    filtered_data.append(item)
                elif operator == "lt" and item_value < value:
                    filtered_data.append(item)
                elif operator == "contains" and isinstance(item_value, str) and value in item_value:
                    filtered_data.append(item)
                elif operator == "regex" and isinstance(item_value, str) and re.search(value, item_value):
                    filtered_data.append(item)
        
        return filtered_data
    
    async def _apply_map(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply map transformation."""
        if not isinstance(data, list):
            raise DataTransformationError("Map requires list data")
        
        field_mappings = config.get("mappings", {})
        
        mapped_data = []
        for item in data:
            if isinstance(item, dict):
                mapped_item = {}
                for old_field, new_field in field_mappings.items():
                    if old_field in item:
                        mapped_item[new_field] = item[old_field]
                mapped_data.append(mapped_item)
            else:
                mapped_data.append(item)
        
        return mapped_data
    
    async def _apply_aggregation(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply aggregation transformation."""
        if not isinstance(data, list):
            raise DataTransformationError("Aggregation requires list data")
        
        agg_type = config.get("type", "count")
        field = config.get("field")
        
        if agg_type == "count":
            return len(data)
        elif agg_type == "sum" and field:
            return sum(item.get(field, 0) for item in data if isinstance(item, dict))
        elif agg_type == "avg" and field:
            values = [item.get(field, 0) for item in data if isinstance(item, dict) and field in item]
            return sum(values) / len(values) if values else 0
        elif agg_type == "min" and field:
            values = [item.get(field) for item in data if isinstance(item, dict) and field in item]
            return min(values) if values else None
        elif agg_type == "max" and field:
            values = [item.get(field) for item in data if isinstance(item, dict) and field in item]
            return max(values) if values else None
        
        return data
    
    async def _apply_sort(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply sort transformation."""
        if not isinstance(data, list):
            raise DataTransformationError("Sort requires list data")
        
        field = config.get("field")
        reverse = config.get("reverse", False)
        
        if field:
            return sorted(data, key=lambda x: x.get(field, 0) if isinstance(x, dict) else x, reverse=reverse)
        else:
            return sorted(data, reverse=reverse)
    
    async def _apply_grouping(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply grouping transformation."""
        if not isinstance(data, list):
            raise DataTransformationError("Grouping requires list data")
        
        field = config.get("field")
        if not field:
            raise DataTransformationError("Grouping requires field specification")
        
        groups = defaultdict(list)
        for item in data:
            if isinstance(item, dict) and field in item:
                key = item[field]
                groups[key].append(item)
        
        return dict(groups)
    
    async def _apply_cleaning(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply data cleaning transformation."""
        operations = config.get("operations", [])
        
        for operation in operations:
            if operation == "remove_nulls":
                data = self._remove_nulls(data)
            elif operation == "trim_strings":
                data = self._trim_strings(data)
            elif operation == "standardize_case":
                case_type = config.get("case", "lower")
                data = self._standardize_case(data, case_type)
            elif operation == "remove_duplicates":
                data = self._remove_duplicates(data)
        
        return data
    
    def _remove_nulls(self, data: Any) -> Any:
        """Remove null/empty values."""
        if isinstance(data, list):
            return [item for item in data if item is not None and item != ""]
        elif isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None and v != ""}
        return data
    
    def _trim_strings(self, data: Any) -> Any:
        """Trim whitespace from strings."""
        if isinstance(data, str):
            return data.strip()
        elif isinstance(data, list):
            return [self._trim_strings(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._trim_strings(v) for k, v in data.items()}
        return data
    
    def _standardize_case(self, data: Any, case_type: str) -> Any:
        """Standardize string case."""
        if isinstance(data, str):
            if case_type == "lower":
                return data.lower()
            elif case_type == "upper":
                return data.upper()
            elif case_type == "title":
                return data.title()
        elif isinstance(data, list):
            return [self._standardize_case(item, case_type) for item in data]
        elif isinstance(data, dict):
            return {k: self._standardize_case(v, case_type) for k, v in data.items()}
        return data
    
    def _remove_duplicates(self, data: Any) -> Any:
        """Remove duplicate items."""
        if isinstance(data, list):
            seen = set()
            result = []
            for item in data:
                # Convert unhashable types to string for comparison
                try:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
                except TypeError:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        result.append(item)
            return result
        return data
    
    async def analyze_data(self, data: Any) -> ProcessingResult:
        """Analyze data structure and content."""
        start_time = datetime.now()
        
        try:
            analysis = {
                "type": type(data).__name__,
                "size": len(data) if hasattr(data, "__len__") else 1,
                "structure": self._analyze_structure(data),
                "statistics": self._calculate_statistics(data),
                "quality": self._assess_data_quality(data)
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=True, processing_time=processing_time)
            
            return ProcessingResult(
                success=True,
                data=analysis,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    def _analyze_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze data structure."""
        if isinstance(data, dict):
            return {
                "type": "object",
                "fields": list(data.keys()),
                "field_count": len(data),
                "field_types": {k: type(v).__name__ for k, v in data.items()}
            }
        elif isinstance(data, list):
            if data:
                first_item = data[0]
                return {
                    "type": "array",
                    "length": len(data),
                    "item_type": type(first_item).__name__,
                    "item_structure": self._analyze_structure(first_item) if isinstance(first_item, (dict, list)) else None
                }
            else:
                return {"type": "array", "length": 0}
        else:
            return {"type": "primitive", "data_type": type(data).__name__}
    
    def _calculate_statistics(self, data: Any) -> Dict[str, Any]:
        """Calculate basic statistics."""
        stats = {}
        
        if isinstance(data, list):
            stats["count"] = len(data)
            
            # Analyze numeric data
            numeric_values = [item for item in data if isinstance(item, (int, float))]
            if numeric_values:
                stats["numeric"] = {
                    "count": len(numeric_values),
                    "mean": np.mean(numeric_values),
                    "median": np.median(numeric_values),
                    "std": np.std(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values)
                }
            
            # Analyze string data
            string_values = [item for item in data if isinstance(item, str)]
            if string_values:
                lengths = [len(s) for s in string_values]
                stats["strings"] = {
                    "count": len(string_values),
                    "avg_length": np.mean(lengths),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "unique_count": len(set(string_values))
                }
        
        elif isinstance(data, dict):
            stats["field_count"] = len(data)
            stats["value_types"] = Counter(type(v).__name__ for v in data.values())
        
        return stats
    
    def _assess_data_quality(self, data: Any) -> Dict[str, Any]:
        """Assess data quality metrics."""
        quality = {
            "completeness": 1.0,
            "consistency": 1.0,
            "accuracy": 1.0,
            "issues": []
        }
        
        if isinstance(data, list):
            if not data:
                quality["issues"].append("Empty dataset")
                quality["completeness"] = 0.0
            else:
                # Check for missing values
                null_count = sum(1 for item in data if item is None or item == "")
                quality["completeness"] = 1.0 - (null_count / len(data))
                
                if null_count > 0:
                    quality["issues"].append(f"{null_count} null/empty values found")
        
        elif isinstance(data, dict):
            # Check for missing values in dict
            null_values = sum(1 for v in data.values() if v is None or v == "")
            if null_values > 0:
                quality["completeness"] = 1.0 - (null_values / len(data))
                quality["issues"].append(f"{null_values} null/empty values in fields")
        
        return quality
    
    def _update_stats(self, success: bool, processing_time: float, data_size: int = 0):
        """Update processing statistics."""
        self.stats["operations_count"] += 1
        if success:
            self.stats["success_count"] += 1
        else:
            self.stats["error_count"] += 1
        
        self.stats["total_processing_time"] += processing_time
        self.stats["data_processed_mb"] += data_size / (1024 * 1024)
    
    async def export_data(self, data: Any, format_type: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Export data to specified format."""
        start_time = datetime.now()
        options = options or {}
        
        try:
            if format_type == "json":
                exported_data = json.dumps(data, indent=options.get("indent", 2), ensure_ascii=False)
            elif format_type == "csv":
                exported_data = await self._export_to_csv(data, options)
            elif format_type == "yaml":
                exported_data = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            elif format_type == "xml":
                exported_data = self._export_to_xml(data, options.get("root_tag", "data"))
            else:
                return ProcessingResult(False, errors=[f"Unsupported export format: {format_type}"])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(success=True, processing_time=processing_time)
            
            return ProcessingResult(
                success=True,
                data=exported_data,
                processing_time=processing_time,
                metadata={"format": format_type, "size": len(exported_data)}
            )
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return ProcessingResult(False, errors=[str(e)])
    
    async def _export_to_csv(self, data: Any, options: Dict[str, Any]) -> str:
        """Export data to CSV format."""
        if not isinstance(data, list):
            raise DataTransformationError("CSV export requires list data")
        
        output = io.StringIO()
        delimiter = options.get("delimiter", ",")
        
        if data and isinstance(data[0], dict):
            # Dictionary data
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
            
            if options.get("include_header", True):
                writer.writeheader()
            
            for row in data:
                writer.writerow(row)
        else:
            # Simple list data
            writer = csv.writer(output, delimiter=delimiter)
            for item in data:
                if isinstance(item, (list, tuple)):
                    writer.writerow(item)
                else:
                    writer.writerow([item])
        
        return output.getvalue()
    
    def _export_to_xml(self, data: Any, root_tag: str) -> str:
        """Export data to XML format."""
        root = ET.Element(root_tag)
        self._dict_to_xml(data, root)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)
    
    def _dict_to_xml(self, data: Any, parent: ET.Element):
        """Convert dictionary to XML elements."""
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, str(key))
                self._dict_to_xml(value, child)
        elif isinstance(data, list):
            for item in data:
                item_elem = ET.SubElement(parent, "item")
                self._dict_to_xml(item, item_elem)
        else:
            parent.text = str(data)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "success_rate": self.stats["success_count"] / max(self.stats["operations_count"], 1),
            "avg_processing_time": self.stats["total_processing_time"] / max(self.stats["operations_count"], 1)
        }
