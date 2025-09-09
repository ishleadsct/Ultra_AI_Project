"""
Ultra AI Project - Analysis Agent

Specialized agent for data analysis, reasoning, problem solving,
and analytical thinking tasks.
"""

import asyncio
import json
import re
import statistics
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math

from .base_agent import BaseAgent, AgentConfig, Task, TaskStatus, AgentCapability
from ..utils.logger import get_logger
from ..utils.helpers import sanitize_string, current_timestamp

logger = get_logger(__name__)

class AnalysisType(Enum):
    """Types of analysis."""
    STATISTICAL = "statistical"
    LOGICAL = "logical"
    COMPARATIVE = "comparative"
    TREND = "trend"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    SENTIMENT = "sentiment"
    TEXTUAL = "textual"
    NUMERICAL = "numerical"
    PATTERN = "pattern"
    RISK = "risk"
    DECISION = "decision"

class DataType(Enum):
    """Types of data for analysis."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXTUAL = "textual"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    MIXED = "mixed"

class ConfidenceLevel(Enum):
    """Confidence levels for analysis results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class AnalysisResult:
    """Structure for analysis results."""
    analysis_type: AnalysisType
    findings: List[str]
    insights: List[str]
    recommendations: List[str]
    confidence_level: ConfidenceLevel
    methodology: str
    limitations: List[str]
    supporting_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataSummary:
    """Summary statistics for datasets."""
    total_records: int
    data_types: Dict[str, DataType]
    missing_values: Dict[str, int]
    summary_stats: Dict[str, Dict[str, float]]
    data_quality_score: float
    anomalies_detected: int

class LogicalReasoning:
    """Logical reasoning framework."""
    
    def __init__(self):
        self.premises = []
        self.conclusions = []
        self.reasoning_steps = []
        self.validity_score = 0.0
    
    def add_premise(self, premise: str, confidence: float = 1.0):
        """Add a premise to the reasoning."""
        self.premises.append({
            "statement": premise,
            "confidence": confidence,
            "type": "premise"
        })
    
    def add_reasoning_step(self, step: str, logic_type: str = "deductive"):
        """Add a reasoning step."""
        self.reasoning_steps.append({
            "step": step,
            "logic_type": logic_type,
            "order": len(self.reasoning_steps) + 1
        })
    
    def draw_conclusion(self, conclusion: str, confidence: float = 0.8):
        """Draw a conclusion from the reasoning."""
        self.conclusions.append({
            "conclusion": conclusion,
            "confidence": confidence,
            "based_on_steps": len(self.reasoning_steps)
        })

class ProblemSolver:
    """Problem solving framework."""
    
    def __init__(self, problem_statement: str):
        self.problem_statement = problem_statement
        self.problem_type = ""
        self.constraints = []
        self.assumptions = []
        self.solution_approaches = []
        self.solutions = []
        self.evaluation_criteria = []
    
    def add_constraint(self, constraint: str):
        """Add a constraint to the problem."""
        self.constraints.append(constraint)
    
    def add_assumption(self, assumption: str, confidence: float = 0.8):
        """Add an assumption."""
        self.assumptions.append({
            "assumption": assumption,
            "confidence": confidence
        })
    
    def add_solution_approach(self, approach: str, complexity: str = "medium"):
        """Add a solution approach."""
        self.solution_approaches.append({
            "approach": approach,
            "complexity": complexity,
            "feasibility": "unknown"
        })

class AnalysisAgent(BaseAgent):
    """Specialized agent for analysis, reasoning, and problem solving."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Set up analysis agent configuration
        if config is None:
            config = {}
        
        agent_config = AgentConfig(
            name=config.get("name", "analysis_agent"),
            agent_type="analysis",
            max_concurrent_tasks=config.get("max_concurrent_tasks", 4),
            timeout=config.get("timeout", 800.0),  # 13 minutes for analysis tasks
            memory_limit=config.get("memory_limit", 1200),
            capabilities=[
                AgentCapability.DATA_ANALYSIS.value,
                AgentCapability.REASONING.value,
                AgentCapability.PROBLEM_SOLVING.value,
                AgentCapability.CLASSIFICATION.value,
                AgentCapability.TEXT_GENERATION.value
            ],
            preferred_models=config.get("preferred_models", ["gpt-4", "claude-3-opus"]),
            enable_memory=config.get("enable_memory", True),
            custom_settings=config.get("custom_settings", {})
        )
        
        super().__init__(agent_config, **kwargs)
        
        # Analysis-specific configuration
        self.analysis_depth = config.get("analysis_depth", "thorough")  # quick, standard, thorough, deep
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.enable_statistical_analysis = config.get("enable_statistical_analysis", True)
        self.max_data_size = config.get("max_data_size", 1000000)  # 1M rows
        
        # Analysis state
        self.active_analyses = {}
        self.reasoning_frameworks = {}
        self.problem_solvers = {}
        self.analysis_templates = {}
        
        # Analysis statistics
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "analysis_types": {},
            "avg_processing_time": 0.0,
            "confidence_distribution": {level.value: 0 for level in ConfidenceLevel}
        }
        
        logger.info("AnalysisAgent initialized")
    
    async def _agent_initialize(self):
        """Analysis agent specific initialization."""
        try:
            # Load analysis templates and methodologies
            await self._load_analysis_templates()
            
            # Initialize statistical methods
            await self._initialize_statistical_methods()
            
            # Load reasoning frameworks
            await self._load_reasoning_frameworks()
            
            # Initialize problem-solving templates
            await self._load_problem_solving_templates()
            
            logger.info("AnalysisAgent initialization complete")
            
        except Exception as e:
            logger.error(f"AnalysisAgent initialization failed: {e}")
            raise
    
    async def _load_analysis_templates(self):
        """Load analysis templates and methodologies."""
        try:
            templates = {
                "statistical_analysis": {
                    "descriptive": {
                        "steps": [
                            "Data collection and validation",
                            "Calculate central tendencies (mean, median, mode)",
                            "Calculate measures of dispersion (std dev, variance)",
                            "Identify outliers and anomalies",
                            "Generate summary statistics",
                            "Create visualizations"
                        ],
                        "outputs": ["summary_stats", "distributions", "outliers"]
                    },
                    "inferential": {
                        "steps": [
                            "Define hypothesis",
                            "Select appropriate test",
                            "Check assumptions",
                            "Perform statistical test",
                            "Interpret results",
                            "Draw conclusions"
                        ],
                        "outputs": ["hypothesis_test", "p_values", "confidence_intervals"]
                    }
                },
                "comparative_analysis": {
                    "steps": [
                        "Define comparison criteria",
                        "Collect comparable data",
                        "Normalize data if needed",
                        "Apply comparison methodology",
                        "Identify patterns and differences",
                        "Rank or score alternatives",
                        "Synthesize findings"
                    ],
                    "methods": ["side_by_side", "weighted_scoring", "gap_analysis"]
                },
                "trend_analysis": {
                    "steps": [
                        "Organize data chronologically",
                        "Identify trend patterns",
                        "Calculate trend metrics",
                        "Analyze seasonality",
                        "Detect change points",
                        "Project future trends",
                        "Assess trend reliability"
                    ],
                    "patterns": ["linear", "exponential", "cyclical", "seasonal"]
                },
                "causal_analysis": {
                    "steps": [
                        "Identify potential causes",
                        "Establish temporal relationships",
                        "Control for confounding variables",
                        "Test causal hypotheses",
                        "Measure effect sizes",
                        "Validate causal claims",
                        "Document limitations"
                    ],
                    "methods": ["correlation", "regression", "experimental", "quasi_experimental"]
                }
            }
            
            await self.store_memory(
                content=templates,
                memory_type="analysis_templates",
                importance=3.5,
                tags=["templates", "methodology", "analysis"]
            )
            
            self.analysis_templates = templates
            
        except Exception as e:
            logger.error(f"Failed to load analysis templates: {e}")
    
    async def _initialize_statistical_methods(self):
        """Initialize statistical analysis methods."""
        try:
            statistical_methods = {
                "descriptive_stats": {
                    "central_tendency": ["mean", "median", "mode"],
                    "dispersion": ["standard_deviation", "variance", "range", "iqr"],
                    "shape": ["skewness", "kurtosis"],
                    "percentiles": [25, 50, 75, 90, 95, 99]
                },
                "hypothesis_tests": {
                    "parametric": ["t_test", "anova", "chi_square"],
                    "non_parametric": ["mann_whitney", "kruskal_wallis", "wilcoxon"],
                    "assumptions": ["normality", "homoscedasticity", "independence"]
                },
                "correlation_analysis": {
                    "methods": ["pearson", "spearman", "kendall"],
                    "interpretation": {
                        "strong": [0.7, 1.0],
                        "moderate": [0.3, 0.7],
                        "weak": [0.1, 0.3],
                        "negligible": [0.0, 0.1]
                    }
                },
                "regression_analysis": {
                    "types": ["linear", "polynomial", "logistic", "multiple"],
                    "diagnostics": ["r_squared", "residuals", "multicollinearity"],
                    "validation": ["cross_validation", "train_test_split"]
                }
            }
            
            await self.store_memory(
                content=statistical_methods,
                memory_type="statistical_methods",
                importance=3.0,
                tags=["statistics", "methods", "analysis"]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize statistical methods: {e}")
    
    async def _load_reasoning_frameworks(self):
        """Load logical reasoning frameworks."""
        try:
            reasoning_frameworks = {
                "deductive_reasoning": {
                    "structure": ["major_premise", "minor_premise", "conclusion"],
                    "validity_criteria": ["logical_form", "premise_truth", "conclusion_follows"],
                    "common_forms": ["modus_ponens", "modus_tollens", "hypothetical_syllogism"]
                },
                "inductive_reasoning": {
                    "types": ["generalization", "analogy", "causal_inference"],
                    "strength_factors": ["sample_size", "representativeness", "consistency"],
                    "limitations": ["hasty_generalization", "weak_analogy", "post_hoc"]
                },
                "abductive_reasoning": {
                    "process": ["observation", "hypothesis_generation", "best_explanation"],
                    "criteria": ["explanatory_power", "simplicity", "consistency"],
                    "applications": ["diagnosis", "theory_formation", "detective_work"]
                },
                "critical_thinking": {
                    "elements": ["purpose", "question", "assumptions", "evidence", "concepts", "implications"],
                    "standards": ["clarity", "accuracy", "precision", "relevance", "depth", "breadth"],
                    "biases": ["confirmation", "anchoring", "availability", "representativeness"]
                }
            }
            
            await self.store_memory(
                content=reasoning_frameworks,
                memory_type="reasoning_frameworks",
                importance=3.0,
                tags=["reasoning", "logic", "critical_thinking"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load reasoning frameworks: {e}")
    
    async def _load_problem_solving_templates(self):
        """Load problem-solving methodologies."""
        try:
            problem_solving = {
                "systematic_approach": {
                    "steps": [
                        "Problem identification and definition",
                        "Information gathering",
                        "Alternative generation",
                        "Alternative evaluation",
                        "Decision making",
                        "Implementation planning",
                        "Solution evaluation"
                    ]
                },
                "root_cause_analysis": {
                    "methods": ["5_whys", "fishbone_diagram", "fault_tree_analysis"],
                    "process": [
                        "Define the problem",
                        "Gather data",
                        "Identify possible causes",
                        "Test hypotheses",
                        "Identify root cause",
                        "Develop solutions"
                    ]
                },
                "decision_analysis": {
                    "frameworks": ["decision_matrix", "cost_benefit", "risk_analysis"],
                    "criteria": ["feasibility", "impact", "cost", "time", "risk"],
                    "weighting_methods": ["equal_weights", "rank_order", "swing_weighting"]
                },
                "creative_problem_solving": {
                    "techniques": ["brainstorming", "mind_mapping", "scamper", "six_thinking_hats"],
                    "phases": ["clarification", "ideation", "development", "implementation"]
                }
            }
            
            await self.store_memory(
                content=problem_solving,
                memory_type="problem_solving",
                importance=3.0,
                tags=["problem_solving", "decision_making", "methodology"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load problem-solving templates: {e}")
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute an analysis task."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "statistical_analysis":
                return await self._statistical_analysis(data)
            elif task_type == "comparative_analysis":
                return await self._comparative_analysis(data)
            elif task_type == "trend_analysis":
                return await self._trend_analysis(data)
            elif task_type == "logical_reasoning":
                return await self._logical_reasoning(data)
            elif task_type == "problem_solving":
                return await self._problem_solving(data)
            elif task_type == "decision_analysis":
                return await self._decision_analysis(data)
            elif task_type == "pattern_analysis":
                return await self._pattern_analysis(data)
            elif task_type == "sentiment_analysis":
                return await self._sentiment_analysis(data)
            elif task_type == "risk_analysis":
                return await self._risk_analysis(data)
            elif task_type == "causal_analysis":
                return await self._causal_analysis(data)
            elif task_type == "predictive_analysis":
                return await self._predictive_analysis(data)
            elif task_type == "data_summary":
                return await self._data_summary(data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _statistical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on data."""
        try:
            dataset = data.get("dataset", [])
            analysis_type = data.get("analysis_type", "descriptive")
            variables = data.get("variables", [])
            hypothesis = data.get("hypothesis", "")
            confidence_level = data.get("confidence_level", 0.95)
            
            if not dataset:
                raise ValueError("Dataset is required for statistical analysis")
            
            # Validate and prepare data
            validated_data = await self._validate_dataset(dataset)
            
            if analysis_type == "descriptive":
                results = await self._descriptive_statistics(validated_data, variables)
            elif analysis_type == "inferential":
                results = await self._inferential_statistics(validated_data, hypothesis, confidence_level)
            elif analysis_type == "correlation":
                results = await self._correlation_analysis(validated_data, variables)
            elif analysis_type == "regression":
                results = await self._regression_analysis(validated_data, variables)
            else:
                raise ValueError(f"Unsupported statistical analysis type: {analysis_type}")
            
            # Generate interpretation
            interpretation = await self._interpret_statistical_results(results, analysis_type)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.STATISTICAL,
                findings=results.get("findings", []),
                insights=interpretation.get("insights", []),
                recommendations=interpretation.get("recommendations", []),
                confidence_level=ConfidenceLevel.HIGH,
                methodology=f"Statistical analysis ({analysis_type})",
                limitations=results.get("limitations", []),
                supporting_data=results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "analysis_type": analysis_type,
                "results": results,
                "interpretation": interpretation,
                "confidence_level": analysis_result.confidence_level.value,
                "methodology": analysis_result.methodology,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.analysis_stats["total_analyses"] += 1
            self.analysis_stats["successful_analyses"] += 1
            self.analysis_stats["analysis_types"][analysis_type] = self.analysis_stats["analysis_types"].get(analysis_type, 0) + 1
            
            return result
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            raise
    
    async def _validate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and prepare dataset for analysis."""
        try:
            if not dataset:
                raise ValueError("Empty dataset")
            
            if len(dataset) > self.max_data_size:
                raise ValueError(f"Dataset too large: {len(dataset)} > {self.max_data_size}")
            
            # Analyze data structure
            if isinstance(dataset[0], dict):
                columns = list(dataset[0].keys())
                data_summary = DataSummary(
                    total_records=len(dataset),
                    data_types={},
                    missing_values={},
                    summary_stats={},
                    data_quality_score=0.0,
                    anomalies_detected=0
                )
                
                # Analyze each column
                for col in columns:
                    values = [row.get(col) for row in dataset]
                    non_null_values = [v for v in values if v is not None]
                    
                    # Determine data type
                    if all(isinstance(v, (int, float)) for v in non_null_values):
                        data_summary.data_types[col] = DataType.NUMERICAL
                        
                        # Calculate summary statistics
                        if non_null_values:
                            data_summary.summary_stats[col] = {
                                "count": len(non_null_values),
                                "mean": statistics.mean(non_null_values),
                                "median": statistics.median(non_null_values),
                                "std_dev": statistics.stdev(non_null_values) if len(non_null_values) > 1 else 0,
                                "min": min(non_null_values),
                                "max": max(non_null_values)
                            }
                    elif all(isinstance(v, bool) for v in non_null_values):
                        data_summary.data_types[col] = DataType.BOOLEAN
                    elif all(isinstance(v, str) for v in non_null_values):
                        data_summary.data_types[col] = DataType.CATEGORICAL
                    else:
                        data_summary.data_types[col] = DataType.MIXED
                    
                    # Count missing values
                    data_summary.missing_values[col] = len(values) - len(non_null_values)
                
                # Calculate data quality score
                total_cells = len(dataset) * len(columns)
                missing_cells = sum(data_summary.missing_values.values())
                data_summary.data_quality_score = (total_cells - missing_cells) / total_cells
                
                return {
                    "data": dataset,
                    "summary": data_summary,
                    "columns": columns,
                    "validated": True
                }
            else:
                # Simple list of values
                return {
                    "data": dataset,
                    "summary": None,
                    "columns": ["value"],
                    "validated": True
                }
                
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            raise
    
    async def _descriptive_statistics(self, validated_data: Dict[str, Any], variables: List[str]) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        try:
            dataset = validated_data["data"]
            summary = validated_data["summary"]
            
            if not variables:
                # Use all numerical columns
                variables = [col for col, dtype in summary.data_types.items() if dtype == DataType.NUMERICAL]
            
            results = {
                "descriptive_stats": {},
                "findings": [],
                "limitations": []
            }
            
            for var in variables:
                if var not in summary.data_types:
                    continue
                
                if summary.data_types[var] == DataType.NUMERICAL:
                    values = [row[var] for row in dataset if row.get(var) is not None]
                    
                    if values:
                        stats = {
                            "count": len(values),
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                            "variance": statistics.variance(values) if len(values) > 1 else 0,
                            "min": min(values),
                            "max": max(values),
                            "range": max(values) - min(values)
                        }
                        
                        # Calculate percentiles
                        sorted_values = sorted(values)
                        n = len(sorted_values)
                        
                        stats["percentiles"] = {
                            "25th": sorted_values[int(n * 0.25)] if n > 0 else None,
                            "75th": sorted_values[int(n * 0.75)] if n > 0 else None,
                            "90th": sorted_values[int(n * 0.90)] if n > 0 else None
                        }
                        
                        # Detect outliers (simple IQR method)
                        q1 = stats["percentiles"]["25th"]
                        q3 = stats["percentiles"]["75th"]
                        if q1 is not None and q3 is not None:
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outliers = [v for v in values if v < lower_bound or v > upper_bound]
                            stats["outliers"] = {
                                "count": len(outliers),
                                "values": outliers[:10]  # Limit to first 10
                            }
                        
                        results["descriptive_stats"][var] = stats
                        
                        # Generate findings
                        if stats["std_dev"] > stats["mean"]:
                            results["findings"].append(f"{var}: High variability (std dev > mean)")
                        
                        if stats["outliers"]["count"] > 0:
                            results["findings"].append(f"{var}: {stats['outliers']['count']} outliers detected")
                
                elif summary.data_types[var] == DataType.CATEGORICAL:
                    values = [row[var] for row in dataset if row.get(var) is not None]
                    value_counts = {}
                    for value in values:
                        value_counts[value] = value_counts.get(value, 0) + 1
                    
                    results["descriptive_stats"][var] = {
                        "count": len(values),
                        "unique_values": len(value_counts),
                        "value_counts": dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                        "most_frequent": max(value_counts.items(), key=lambda x: x[1]) if value_counts else None
                    }
            
            # Add limitations
            if summary.data_quality_score < 0.9:
                results["limitations"].append(f"Data quality concerns: {summary.data_quality_score:.1%} completeness")
            
            if len(dataset) < 30:
                results["limitations"].append("Small sample size may limit statistical validity")
            
            return results
            
        except Exception as e:
            logger.error(f"Descriptive statistics calculation failed: {e}")
            raise
    
    async def _inferential_statistics(self, validated_data: Dict[str, Any], 
                                    hypothesis: str, confidence_level: float) -> Dict[str, Any]:
        """Perform inferential statistical analysis."""
        try:
            # This is a simplified implementation
            # In practice, you would implement specific statistical tests
            
            results = {
                "hypothesis": hypothesis,
                "confidence_level": confidence_level,
                "test_results": {},
                "findings": [],
                "limitations": ["Simplified implementation", "Manual test selection needed"]
            }
            
            dataset = validated_data["data"]
            summary = validated_data["summary"]
            
            # Example: Basic normality testing for numerical variables
            numerical_vars = [col for col, dtype in summary.data_types.items() if dtype == DataType.NUMERICAL]
            
            for var in numerical_vars:
                values = [row[var] for row in dataset if row.get(var) is not None]
                
                if len(values) >= 30:
                    # Simple normality check using mean and median comparison
                    mean_val = statistics.mean(values)
                    median_val = statistics.median(values)
                    
                    # Calculate skewness approximation
                    mean_median_diff = abs(mean_val - median_val)
                    std_dev = statistics.stdev(values) if len(values) > 1 else 0
                    
                    normality_indicator = mean_median_diff / std_dev if std_dev > 0 else 0
                    
                    results["test_results"][var] = {
                        "normality_check": {
                            "mean_median_difference": mean_median_diff,
                            "relative_difference": normality_indicator,
                            "assessment": "approximately_normal" if normality_indicator < 0.5 else "potentially_skewed"
                        }
                    }
                    
                    if normality_indicator < 0.5:
                        results["findings"].append(f"{var}: Data appears approximately normal")
                    else:
                        results["findings"].append(f"{var}: Data may be skewed (consider non-parametric tests)")
            
            return results
            
        except Exception as e:
            logger.error(f"Inferential statistics failed: {e}")
            raise
    
    async def _correlation_analysis(self, validated_data: Dict[str, Any], variables: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis."""
        try:
            dataset = validated_data["data"]
            summary = validated_data["summary"]
            
            if not variables:
                variables = [col for col, dtype in summary.data_types.items() if dtype == DataType.NUMERICAL]
            
            if len(variables) < 2:
                raise ValueError("At least two numerical variables required for correlation analysis")
            
            results = {
                "correlations": {},
                "findings": [],
                "limitations": []
            }
            
            # Calculate pairwise correlations
            for i, var1 in enumerate(variables):
                for var2 in variables[i+1:]:
                    values1 = [row[var1] for row in dataset if row.get(var1) is not None and row.get(var2) is not None]
                    values2 = [row[var2] for row in dataset if row.get(var1) is not None and row.get(var2) is not None]
                    
                    if len(values1) >= 3:  # Minimum for correlation
                        # Calculate Pearson correlation coefficient
                        corr_coef = self._calculate_correlation(values1, values2)
                        
                        pair_key = f"{var1}_vs_{var2}"
                        results["correlations"][pair_key] = {
                            "correlation_coefficient": corr_coef,
                            "strength": self._interpret_correlation_strength(abs(corr_coef)),
                            "direction": "positive" if corr_coef > 0 else "negative" if corr_coef < 0 else "none",
                            "sample_size": len(values1)
                        }
                        
                        # Generate findings
                        if abs(corr_coef) > 0.7:
                            results["findings"].append(f"Strong correlation between {var1} and {var2}: {corr_coef:.3f}")
                        elif abs(corr_coef) > 0.3:
                            results["findings"].append(f"Moderate correlation between {var1} and {var2}: {corr_coef:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        try:
            n = len(x)
            if n != len(y) or n < 2:
                return 0.0
            
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
            
denominator = math.sqrt(sum_sq_x * sum_sq_y)
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.0
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        elif abs_corr >= 0.1:
            return "weak"
        else:
            return "negligible"
    
    async def _regression_analysis(self, validated_data: Dict[str, Any], variables: List[str]) -> Dict[str, Any]:
        """Perform regression analysis."""
        try:
            # Simplified linear regression implementation
            dataset = validated_data["data"]
            
            if len(variables) < 2:
                raise ValueError("At least two variables required for regression (dependent and independent)")
            
            dependent_var = variables[0]
            independent_vars = variables[1:]
            
            # Simple linear regression with first independent variable
            independent_var = independent_vars[0]
            
            # Extract paired values
            pairs = [(row[independent_var], row[dependent_var]) 
                    for row in dataset 
                    if row.get(independent_var) is not None and row.get(dependent_var) is not None]
            
            if len(pairs) < 3:
                raise ValueError("Insufficient data points for regression")
            
            x_values = [pair[0] for pair in pairs]
            y_values = [pair[1] for pair in pairs]
            
            # Calculate linear regression parameters
            n = len(pairs)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in pairs)
            sum_x_sq = sum(x * x for x in x_values)
            
            # Calculate slope and intercept
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pairs)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            results = {
                "regression_equation": f"y = {slope:.4f}x + {intercept:.4f}",
                "coefficients": {
                    "slope": slope,
                    "intercept": intercept
                },
                "model_fit": {
                    "r_squared": r_squared,
                    "interpretation": self._interpret_r_squared(r_squared)
                },
                "sample_size": n,
                "dependent_variable": dependent_var,
                "independent_variable": independent_var,
                "findings": [],
                "limitations": ["Simple linear regression", "Single independent variable"]
            }
            
            # Generate findings
            if r_squared > 0.7:
                results["findings"].append(f"Strong predictive relationship (R² = {r_squared:.3f})")
            elif r_squared > 0.3:
                results["findings"].append(f"Moderate predictive relationship (R² = {r_squared:.3f})")
            else:
                results["findings"].append(f"Weak predictive relationship (R² = {r_squared:.3f})")
            
            if slope > 0:
                results["findings"].append(f"Positive relationship: {dependent_var} increases with {independent_var}")
            else:
                results["findings"].append(f"Negative relationship: {dependent_var} decreases with {independent_var}")
            
            return results
            
        except Exception as e:
            logger.error(f"Regression analysis failed: {e}")
            raise
    
    def _interpret_r_squared(self, r_squared: float) -> str:
        """Interpret R-squared value."""
        if r_squared >= 0.7:
            return "strong_fit"
        elif r_squared >= 0.3:
            return "moderate_fit"
        else:
            return "weak_fit"
    
    async def _interpret_statistical_results(self, results: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Generate interpretation of statistical results."""
        try:
            prompt = f"""Interpret the following {analysis_type} statistical analysis results:

{json.dumps(results, indent=2, default=str)}

Please provide:
1. Key insights from the analysis
2. Practical implications of the findings
3. Recommendations for action or further analysis
4. Limitations and caveats to consider

Focus on making the statistical results understandable and actionable."""
            
            if not self.model_manager:
                return {
                    "insights": ["Statistical interpretation not available"],
                    "recommendations": ["Manual interpretation required"]
                }
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            if response.success:
                # Parse interpretation from response
                interpretation = self._parse_interpretation_response(response.content)
                return interpretation
            else:
                return {
                    "insights": ["Interpretation generation failed"],
                    "recommendations": ["Manual analysis required"]
                }
                
        except Exception as e:
            logger.error(f"Statistical interpretation failed: {e}")
            return {
                "insights": [f"Interpretation error: {str(e)}"],
                "recommendations": ["Manual interpretation required"]
            }
    
    def _parse_interpretation_response(self, response: str) -> Dict[str, List[str]]:
        """Parse interpretation response into structured format."""
        interpretation = {
            "insights": [],
            "implications": [],
            "recommendations": [],
            "limitations": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify sections
            line_lower = line.lower()
            if 'insights' in line_lower or 'findings' in line_lower:
                current_section = 'insights'
            elif 'implications' in line_lower or 'practical' in line_lower:
                current_section = 'implications'
            elif 'recommendations' in line_lower or 'action' in line_lower:
                current_section = 'recommendations'
            elif 'limitations' in line_lower or 'caveats' in line_lower:
                current_section = 'limitations'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                # List item
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item:
                    interpretation[current_section].append(item)
        
        return interpretation
    
    async def _comparative_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis."""
        try:
            items = data.get("items", [])
            criteria = data.get("criteria", [])
            weights = data.get("weights", {})
            comparison_type = data.get("comparison_type", "qualitative")
            
            if len(items) < 2:
                raise ValueError("At least two items required for comparison")
            
            if not criteria:
                raise ValueError("Comparison criteria must be specified")
            
            # Build comparison prompt
            comparison_prompt = f"""Perform a {comparison_type} comparative analysis of the following items:

Items to compare:
{json.dumps(items, indent=2)}

Comparison criteria:
{', '.join(criteria)}

Weights (if any):
{json.dumps(weights, indent=2)}

Please provide:
1. Side-by-side comparison across all criteria
2. Strengths and weaknesses of each item
3. Overall ranking or scoring
4. Key differentiators
5. Recommendations based on the comparison

Be objective and thorough in your analysis."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=comparison_prompt,
                temperature=0.2,
                max_tokens=1200
            )
            
            if not response.success:
                raise ValueError(f"Comparative analysis failed: {response.error}")
            
            # Parse comparison results
            comparison_results = self._parse_comparison_response(response.content, items, criteria)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.COMPARATIVE,
                findings=comparison_results.get("findings", []),
                insights=comparison_results.get("insights", []),
                recommendations=comparison_results.get("recommendations", []),
                confidence_level=ConfidenceLevel.HIGH,
                methodology="Structured comparative analysis",
                limitations=["Qualitative assessment", "Subjective criteria weighting"],
                supporting_data=comparison_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "comparison_type": comparison_type,
                "items": items,
                "criteria": criteria,
                "comparison_results": comparison_results,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            raise
    
    def _parse_comparison_response(self, response: str, items: List[str], criteria: List[str]) -> Dict[str, Any]:
        """Parse comparison response into structured format."""
        results = {
            "detailed_comparison": response,
            "findings": [],
            "insights": [],
            "recommendations": [],
            "scores": {},
            "ranking": []
        }
        
        # Simple parsing - could be enhanced with more sophisticated NLP
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify sections and extract information
            line_lower = line.lower()
            if 'findings' in line_lower or 'key differences' in line_lower:
                current_section = 'findings'
            elif 'insights' in line_lower or 'analysis' in line_lower:
                current_section = 'insights'
            elif 'recommendations' in line_lower:
                current_section = 'recommendations'
            elif 'ranking' in line_lower or 'order' in line_lower:
                current_section = 'ranking'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    if current_section == 'ranking':
                        results['ranking'].append(item)
                    else:
                        results[current_section].append(item)
        
        return results

    async def _trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trend analysis on time-series data."""
        try:
            time_series_data = data.get("time_series_data", [])
            time_column = data.get("time_column", "date")
            value_column = data.get("value_column", "value")
            trend_type = data.get("trend_type", "linear")
            
            if not time_series_data:
                raise ValueError("Time series data is required")
            
            # Extract time and value arrays
            time_values = []
            data_values = []
            
            for point in time_series_data:
                if time_column in point and value_column in point:
                    time_values.append(point[time_column])
                    data_values.append(point[value_column])
            
            if len(data_values) < 3:
                raise ValueError("Insufficient data points for trend analysis")
            
            # Calculate trend metrics
            trend_results = await self._calculate_trend_metrics(data_values, trend_type)
            
            # Analyze patterns
            patterns = await self._analyze_patterns(data_values)
            
            # Generate trend interpretation
            interpretation = await self._interpret_trend_results(trend_results, patterns)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.TREND,
                findings=trend_results.get("findings", []),
                insights=interpretation.get("insights", []),
                recommendations=interpretation.get("recommendations", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology=f"Trend analysis ({trend_type})",
                limitations=["Limited historical data", "Assumes trend continuation"],
                supporting_data=trend_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "trend_type": trend_type,
                "trend_results": trend_results,
                "patterns": patterns,
                "interpretation": interpretation,
                "data_points": len(data_values),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise
    
    async def _calculate_trend_metrics(self, data_values: List[float], trend_type: str) -> Dict[str, Any]:
        """Calculate trend metrics."""
        try:
            n = len(data_values)
            
            # Basic trend metrics
            first_value = data_values[0]
            last_value = data_values[-1]
            total_change = last_value - first_value
            percent_change = (total_change / first_value * 100) if first_value != 0 else 0
            
            # Calculate moving averages
            if n >= 3:
                moving_avg_3 = [sum(data_values[i:i+3])/3 for i in range(n-2)]
            else:
                moving_avg_3 = []
            
            # Simple linear trend
            x_values = list(range(n))
            if n >= 2:
                # Calculate slope
                sum_x = sum(x_values)
                sum_y = sum(data_values)
                sum_xy = sum(x * y for x, y in zip(x_values, data_values))
                sum_x_sq = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x)
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            else:
                slope = 0
                trend_direction = "stable"
            
            # Volatility measure (standard deviation)
            mean_value = sum(data_values) / n
            variance = sum((x - mean_value) ** 2 for x in data_values) / n
            volatility = math.sqrt(variance)
            
            results = {
                "trend_direction": trend_direction,
                "slope": slope,
                "total_change": total_change,
                "percent_change": percent_change,
                "volatility": volatility,
                "mean_value": mean_value,
                "min_value": min(data_values),
                "max_value": max(data_values),
                "moving_averages": {
                    "3_period": moving_avg_3[-1] if moving_avg_3 else None
                },
                "findings": []
            }
            
            # Generate findings
            if abs(percent_change) > 20:
                results["findings"].append(f"Significant change: {percent_change:.1f}%")
            
            if volatility > mean_value * 0.2:
                results["findings"].append("High volatility detected")
            
            if trend_direction != "stable":
                results["findings"].append(f"Clear {trend_direction} trend (slope: {slope:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Trend metrics calculation failed: {e}")
            raise
    
    async def _analyze_patterns(self, data_values: List[float]) -> Dict[str, Any]:
        """Analyze patterns in the data."""
        try:
            patterns = {
                "cyclical": False,
                "seasonal": False,
                "outliers": [],
                "change_points": [],
                "stability_periods": []
            }
            
            n = len(data_values)
            if n < 4:
                return patterns
            
            # Simple outlier detection
            mean_val = sum(data_values) / n
            std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in data_values) / n)
            threshold = 2 * std_dev
            
            for i, value in enumerate(data_values):
                if abs(value - mean_val) > threshold:
                    patterns["outliers"].append({
                        "index": i,
                        "value": value,
                        "deviation": abs(value - mean_val)
                    })
            
            # Simple change point detection (significant slope changes)
            if n >= 6:
                for i in range(2, n-2):
                    # Compare slopes before and after point i
                    before_slope = (data_values[i] - data_values[i-2]) / 2
                    after_slope = (data_values[i+2] - data_values[i]) / 2
                    
                    if abs(before_slope - after_slope) > std_dev:
                        patterns["change_points"].append({
                            "index": i,
                            "value": data_values[i],
                            "slope_change": after_slope - before_slope
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {"cyclical": False, "seasonal": False, "outliers": [], "change_points": []}
    
    async def _interpret_trend_results(self, trend_results: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret trend analysis results."""
        try:
            prompt = f"""Interpret the following trend analysis results:

Trend Results:
{json.dumps(trend_results, indent=2, default=str)}

Patterns Detected:
{json.dumps(patterns, indent=2, default=str)}

Please provide:
1. Key insights about the trend
2. What the trend indicates about the underlying process
3. Potential causes of observed patterns
4. Future projections and implications
5. Recommendations for monitoring or action

Focus on practical implications and actionable insights."""
            
            if not self.model_manager:
                return {
                    "insights": ["Trend interpretation not available"],
                    "recommendations": ["Manual interpretation required"]
                }
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            if response.success:
                return self._parse_interpretation_response(response.content)
            else:
                return {
                    "insights": ["Interpretation generation failed"],
                    "recommendations": ["Manual analysis required"]
                }
                
        except Exception as e:
            logger.error(f"Trend interpretation failed: {e}")
            return {
                "insights": [f"Interpretation error: {str(e)}"],
                "recommendations": ["Manual interpretation required"]
            }
    
    async def _logical_reasoning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform logical reasoning analysis."""
        try:
            premises = data.get("premises", [])
            question = data.get("question", "")
            reasoning_type = data.get("reasoning_type", "deductive")
            
            if not premises:
                raise ValueError("Premises are required for logical reasoning")
            
            # Create reasoning framework
            reasoning = LogicalReasoning()
            
            # Add premises
            for premise in premises:
                confidence = premise.get("confidence", 1.0) if isinstance(premise, dict) else 1.0
                statement = premise.get("statement", premise) if isinstance(premise, dict) else premise
                reasoning.add_premise(statement, confidence)
            
            # Build reasoning prompt
            reasoning_prompt = f"""Perform {reasoning_type} logical reasoning based on the following:

Premises:
{chr(10).join([f"- {p['statement']} (confidence: {p['confidence']})" for p in reasoning.premises])}

Question/Goal: {question}

Please:
1. Analyze the logical structure of the premises
2. Identify any logical relationships or patterns
3. Apply {reasoning_type} reasoning to reach conclusions
4. Assess the validity and soundness of the reasoning
5. Identify any logical fallacies or weaknesses
6. Provide step-by-step reasoning process

Be rigorous and explicit in your logical analysis."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=reasoning_prompt,
                temperature=0.1,  # Low temperature for logical reasoning
                max_tokens=1000
            )
            
            if not response.success:
                raise ValueError(f"Logical reasoning failed: {response.error}")
            
            # Parse reasoning results
            reasoning_results = self._parse_reasoning_response(response.content)
            
            # Calculate validity score
            validity_score = self._calculate_reasoning_validity(reasoning, reasoning_results)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.LOGICAL,
                findings=reasoning_results.get("conclusions", []),
                insights=reasoning_results.get("insights", []),
                recommendations=reasoning_results.get("recommendations", []),
                confidence_level=ConfidenceLevel.HIGH if validity_score > 0.8 else ConfidenceLevel.MEDIUM,
                methodology=f"Logical reasoning ({reasoning_type})",
                limitations=reasoning_results.get("limitations", []),
                supporting_data=reasoning_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "reasoning_type": reasoning_type,
                "premises": premises,
                "question": question,
                "reasoning_results": reasoning_results,
                "validity_score": validity_score,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Logical reasoning failed: {e}")
            raise
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse logical reasoning response."""
        results = {
            "reasoning_steps": [],
            "conclusions": [],
            "insights": [],
            "limitations": [],
            "logical_structure": "",
            "validity_assessment": ""
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'steps' in line_lower or 'reasoning' in line_lower:
                current_section = 'reasoning_steps'
            elif 'conclusions' in line_lower:
                current_section = 'conclusions'
            elif 'insights' in line_lower or 'analysis' in line_lower:
                current_section = 'insights'
            elif 'limitations' in line_lower or 'weaknesses' in line_lower:
                current_section = 'limitations'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    results[current_section].append(item)
        
        return results
    
    def _calculate_reasoning_validity(self, reasoning: LogicalReasoning, results: Dict[str, Any]) -> float:
        """Calculate validity score for reasoning."""
        try:
            # Simple validity assessment based on:
            # 1. Premise confidence
            # 2. Number of reasoning steps
            # 3. Identified limitations
            
            premise_confidence = sum(p["confidence"] for p in reasoning.premises) / len(reasoning.premises)
            step_completeness = min(len(results.get("reasoning_steps", [])) / 3, 1.0)  # Expect at least 3 steps
            limitation_penalty = len(results.get("limitations", [])) * 0.1
            
            validity_score = (premise_confidence * 0.5 + step_completeness * 0.3 + 0.2) - limitation_penalty
            
            return max(0.0, min(1.0, validity_score))
            
        except Exception as e:
            logger.error(f"Validity calculation failed: {e}")
            return 0.5  # Default medium confidence
    
    async def _problem_solving(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform structured problem solving."""
        try:
            problem_statement = data.get("problem_statement", "")
            constraints = data.get("constraints", [])
            objectives = data.get("objectives", [])
            resources = data.get("resources", [])
            approach = data.get("approach", "systematic")
            
            if not problem_statement:
                raise ValueError("Problem statement is required")
            
            # Create problem solver
            solver = ProblemSolver(problem_statement)
            
            # Add constraints
            for constraint in constraints:
                solver.add_constraint(constraint)
            
            # Build problem-solving prompt
            problem_prompt = f"""Solve the following problem using a {approach} approach:

Problem Statement: {problem_statement}

Constraints:
{chr(10).join([f"- {c}" for c in constraints])}

Objectives:
{chr(10).join([f"- {o}" for o in objectives])}

Available Resources:
{chr(10).join([f"- {r}" for r in resources])}

Please provide:
1. Problem analysis and breakdown
2. Alternative solution approaches
3. Evaluation of each approach
4. Recommended solution with rationale
5. Implementation steps
6. Risk assessment and mitigation
7. Success metrics

Use systematic problem-solving methodology."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=problem_prompt,
                temperature=0.3,
                max_tokens=1500
            )
            
            if not response.success:
                raise ValueError(f"Problem solving failed: {response.error}")
            
            # Parse problem-solving results
            solution_results = self._parse_problem_solving_response(response.content)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.DECISION,
                findings=solution_results.get("solutions", []),
                insights=solution_results.get("insights", []),
                recommendations=solution_results.get("recommendations", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology=f"Problem solving ({approach})",
                limitations=solution_results.get("limitations", []),
                supporting_data=solution_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "problem_statement": problem_statement,
                "approach": approach,
                "solution_results": solution_results,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Problem solving failed: {e}")
            raise
    
    def _parse_problem_solving_response(self, response: str) -> Dict[str, Any]:
        """Parse problem-solving response."""
        results = {
            "problem_analysis": "",
            "solutions": [],
            "recommendations": [],
            "implementation_steps": [],
            "risks": [],
            "success_metrics": [],
            "insights": [],
            "limitations": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'analysis' in line_lower or 'breakdown' in line_lower:
                current_section = 'problem_analysis'
            elif 'solutions' in line_lower or 'approaches' in line_lower:
                current_section = 'solutions'
            elif 'recommendations' in line_lower or 'recommended' in line_lower:
                current_section = 'recommendations'
            elif 'implementation' in line_lower or 'steps' in line_lower:
                current_section = 'implementation_steps'
            elif 'risks' in line_lower or 'mitigation' in line_lower:
                current_section = 'risks'
            elif 'metrics' in line_lower or 'success' in line_lower:
                current_section = 'success_metrics'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results and isinstance(results[current_section], list):
                    results[current_section].append(item)
            elif current_section == 'problem_analysis' and not line.startswith(('#', '##')):
                results[current_section] += line + " "
        
        return results
    
    async def _decision_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform decision analysis."""
        try:
            decision_options = data.get("options", [])
            criteria = data.get("criteria", [])
            weights = data.get("weights", {})
            decision_context = data.get("context", "")
            
            if not decision_options:
                raise ValueError("Decision options are required")
            
            if not criteria:
                raise ValueError("Decision criteria are required")
            
            # Build decision analysis prompt
            decision_prompt = f"""Perform a comprehensive decision analysis:

Decision Context: {decision_context}

Options to evaluate:
{json.dumps(decision_options, indent=2)}

Evaluation Criteria:
{chr(10).join([f"- {c}" for c in criteria])}

Criteria Weights:
{json.dumps(weights, indent=2)}

Please provide:
1. Decision matrix with scores for each option against each criterion
2. Weighted scoring analysis
3. Pros and cons for each option
4. Risk assessment for each option
5. Sensitivity analysis
6. Final recommendation with rationale
7. Implementation considerations

Use a structured decision-making framework."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=decision_prompt,
                temperature=0.2,
                max_tokens=1200
            )
            
            if not response.success:
                raise ValueError(f"Decision analysis failed: {response.error}")
            
            # Parse decision results
            decision_results = self._parse_decision_response(response.content, decision_options, criteria)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.DECISION,
                findings=decision_results.get("findings", []),
                insights=decision_results.get("insights", []),
                recommendations=decision_results.get("recommendations", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology="Multi-criteria decision analysis",
                limitations=["Subjective scoring", "Weight assumptions"],
                supporting_data=decision_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "decision_context": decision_context,
                "options": decision_options,
                "criteria": criteria,
                "weights": weights,
                "decision_results": decision_results,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Decision analysis failed: {e}")
            raise
    
    def _parse_decision_response(self, response: str, options: List[str], criteria: List[str]) -> Dict[str, Any]:
        """Parse decision analysis response."""
        results = {
            "decision_matrix": {},
            "rankings": [],
            "recommendations": [],
            "findings": [],
            "insights": [],
            "pros_cons": {},
            "risk_assessment": {}
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'matrix' in line_lower or 'scores' in line_lower:
                current_section = 'decision_matrix'
            elif 'ranking' in line_lower or 'order' in line_lower:
                current_section = 'rankings'
            elif 'recommendations' in line_lower:
                current_section = 'recommendations'
            elif 'findings' in line_lower:
                current_section = 'findings'
            elif 'insights' in line_lower:
                current_section = 'insights'
            elif 'pros' in line_lower and 'cons' in line_lower:
                current_section = 'pros_cons'
            elif 'risk' in line_lower:
                current_section = 'risk_assessment'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    if isinstance(results[current_section], list):
                        results[current_section].append(item)
                    elif isinstance(results[current_section], dict):
                        # Simple parsing for dict sections
                        results[current_section][f"item_{len(results[current_section])}"] = item
        
        return results
    
    async def _pattern_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in data."""
        try:
            dataset = data.get("dataset", [])
            pattern_types = data.get("pattern_types", ["frequency", "sequence", "anomaly"])
            analysis_depth = data.get("analysis_depth", "standard")
            
            if not dataset:
                raise ValueError("Dataset is required for pattern analysis")
            
            # Analyze different pattern types
            pattern_results = {}
            
            if "frequency" in pattern_types:
                pattern_results["frequency"] = await self._analyze_frequency_patterns(dataset)
            
            if "sequence" in pattern_types:
                pattern_results["sequence"] = await self._analyze_sequence_patterns(dataset)
            
            if "anomaly" in pattern_types:
                pattern_results["anomaly"] = await self._analyze_anomaly_patterns(dataset)
            
            if "correlation" in pattern_types:
                pattern_results["correlation"] = await self._analyze_correlation_patterns(dataset)
            
            # Generate pattern insights
            insights = await self._generate_pattern_insights(pattern_results)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.PATTERN,
                findings=pattern_results.get("summary", []),
                insights=insights.get("insights", []),
                recommendations=insights.get("recommendations", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology="Multi-dimensional pattern analysis",
                limitations=["Limited pattern types", "Statistical approximations"],
                supporting_data=pattern_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "pattern_types": pattern_types,
                "pattern_results": pattern_results,
                "insights": insights,
                "data_size": len(dataset),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            raise
    
    async def _analyze_frequency_patterns(self, dataset: List[Any]) -> Dict[str, Any]:
        """Analyze frequency patterns in data."""
        try:
            frequency_map = {}
            
            for item in dataset:
                item_str = str(item)
                frequency_map[item_str] = frequency_map.get(item_str, 0) + 1
            
            # Sort by frequency
            sorted_frequencies = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate statistics
            frequencies = list(frequency_map.values())
            total_items = len(dataset)
            unique_items = len(frequency_map)
            
            results = {
                "total_items": total_items,
                "unique_items": unique_items,
                "diversity_ratio": unique_items / total_items,
                "most_frequent": sorted_frequencies[:10],
                "frequency_distribution": {
                    "mean": statistics.mean(frequencies),
                    "median": statistics.median(frequencies),
                    "std_dev": statistics.stdev(frequencies) if len(frequencies) > 1 else 0
                },
                "patterns": []
            }
            
            # Identify patterns
            if results["diversity_ratio"] < 0.1:
                results["patterns"].append("Low diversity - few items dominate")
            elif results["diversity_ratio"] > 0.8:
                results["patterns"].append("High diversity - items are mostly unique")
            
            # Power law distribution check
            if len(sorted_frequencies) >= 10:
                top_1_percent = sorted_frequencies[0][1] / total_items
                if top_1_percent > 0.1:
                    results["patterns"].append("Potential power law distribution")
            
            return results
            
        except Exception as e:
            logger.error(f"Frequency pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_sequence_patterns(self, dataset: List[Any]) -> Dict[str, Any]:
        """Analyze sequence patterns in data."""
        try:
            if len(dataset) < 3:
                return {"error": "Insufficient data for sequence analysis"}
            
            # Look for repeating subsequences
            patterns = {}
            sequence_str = ''.join(str(item) for item in dataset)
            
            # Check for patterns of different lengths
            for pattern_length in range(2, min(10, len(dataset) // 2)):
                for i in range(len(dataset) - pattern_length + 1):
                    pattern = tuple(dataset[i:i + pattern_length])
                    pattern_str = str(pattern)
                    
                    if pattern_str not in patterns:
                        patterns[pattern_str] = {"pattern": pattern, "positions": [], "count": 0}
                    
                    patterns[pattern_str]["positions"].append(i)
                    patterns[pattern_str]["count"] += 1
            
            # Filter for actual repeating patterns
            repeating_patterns = {k: v for k, v in patterns.items() if v["count"] > 1}
            
            # Sort by frequency and length
            sorted_patterns = sorted(
                repeating_patterns.items(),
                key=lambda x: (x[1]["count"], len(x[1]["pattern"])),
                reverse=True
            )
            
            results = {
                "total_patterns_found": len(repeating_patterns),
                "most_frequent_patterns": sorted_patterns[:10],
                "sequence_length": len(dataset),
                "pattern_coverage": sum(p["count"] for p in repeating_patterns.values()) / len(dataset),
                "insights": []
            }
            
            if results["pattern_coverage"] > 0.5:
                results["insights"].append("High pattern coverage - sequence is highly structured")
            elif results["pattern_coverage"] < 0.1:
                results["insights"].append("Low pattern coverage - sequence appears random")
            
            return results
            
        except Exception as e:
            logger.error(f"Sequence pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_anomaly_patterns(self, dataset: List[Any]) -> Dict[str, Any]:
        """Analyze anomaly patterns in data."""
        try:
            # Convert to numerical if possible
            numerical_data = []
            non_numerical_data = []
            
            for item in dataset:
                try:
                    if isinstance(item, (int, float)):
                        numerical_data.append(float(item))
                    elif isinstance(item, str) and item.replace('.', '').replace('-', '').isdigit():
                        numerical_data.append(float(item))
                    else:
                        non_numerical_data.append(item)
                except:
                    non_numerical_data.append(item)
            
            results = {
                "numerical_anomalies": {},
                "categorical_anomalies": {},
                "total_anomalies": 0
            }
            
            # Numerical anomaly detection
            if len(numerical_data) >= 3:
                mean_val = statistics.mean(numerical_data)
                std_dev = statistics.stdev(numerical_data) if len(numerical_data) > 1 else 0
                
                # Z-score based anomaly detection
                threshold = 2.0  # 2 standard deviations
                anomalies = []
                
                for i, value in enumerate(numerical_data):
                    if std_dev > 0:
                        z_score = abs(value - mean_val) / std_dev
                        if z_score > threshold:
                            anomalies.append({
                                "index": i,
                                "value": value,
                                "z_score": z_score,
                                "deviation": abs(value - mean_val)
                            })
                
                results["numerical_anomalies"] = {
                    "count": len(anomalies),
                    "anomalies": anomalies[:20],  # Limit to first 20
                    "threshold": threshold,
                    "mean": mean_val,
                    "std_dev": std_dev
                }
                
                results["total_anomalies"] += len(anomalies)
            
            # Categorical anomaly detection (rare items)
            if non_numerical_data:
                frequency_map = {}
                for item in non_numerical_data:
                    item_str = str(item)
                    frequency_map[item_str] = frequency_map.get(item_str, 0) + 1
                
                total_items = len(non_numerical_data)
                rare_threshold = 0.05  # Items appearing in less than 5% of data
                
                rare_items = []
                for item, count in frequency_map.items():
                    frequency = count / total_items
                    if frequency < rare_threshold:
                        rare_items.append({
                            "item": item,
                            "count": count,
                            "frequency": frequency
                        })
                
                results["categorical_anomalies"] = {
                    "rare_items": sorted(rare_items, key=lambda x: x["frequency"])[:20],
                    "rare_threshold": rare_threshold,
                    "total_unique_items": len(frequency_map)
                }
                
                results["total_anomalies"] += len(rare_items)
            
            return results
            
        except Exception as e:
            logger.error(f"Anomaly pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_correlation_patterns(self, dataset: List[Any]) -> Dict[str, Any]:
        """Analyze correlation patterns in structured data."""
        try:
            if not dataset or not isinstance(dataset[0], dict):
                return {"error": "Structured data (list of dictionaries) required for correlation analysis"}
            
            # Extract numerical columns
            numerical_columns = {}
            for row in dataset:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        if key not in numerical_columns:
                            numerical_columns[key] = []
                        numerical_columns[key].append(value)
            
            # Calculate correlations between numerical columns
            correlations = {}
            column_names = list(numerical_columns.keys())
            
            for i, col1 in enumerate(column_names):
                for col2 in column_names[i+1:]:
                    # Ensure same length
                    min_length = min(len(numerical_columns[col1]), len(numerical_columns[col2]))
                    values1 = numerical_columns[col1][:min_length]
                    values2 = numerical_columns[col2][:min_length]
                    
                    if len(values1) >= 3:
                        corr_coef = self._calculate_correlation(values1, values2)
                        correlations[f"{col1}_vs_{col2}"] = {
                            "correlation": corr_coef,
                            "strength": self._interpret_correlation_strength(abs(corr_coef)),
                            "sample_size": len(values1)
                        }
            
            # Identify strong correlations
            strong_correlations = {
                k: v for k, v in correlations.items() 
                if abs(v["correlation"]) > 0.5
            }
            
            results = {
                "numerical_columns": len(numerical_columns),
                "correlations": correlations,
                "strong_correlations": strong_correlations,
                "insights": []
            }
            
            if strong_correlations:
                results["insights"].append(f"Found {len(strong_correlations)} strong correlations")
            
            return results
            
        except Exception as e:
            logger.error(f"Correlation pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _generate_pattern_insights(self, pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from pattern analysis results."""
        try:
            prompt = f"""Analyze the following pattern analysis results and provide insights:

Pattern Analysis Results:
{json.dumps(pattern_results, indent=2, default=str)}

Please provide:
1. Key patterns and their significance
2. Insights about data structure and behavior
3. Potential underlying processes or causes
4. Anomalies and their potential explanations
5. Recommendations for further analysis
6. Practical implications of the patterns

Focus on actionable insights and meaningful interpretations."""
            
            if not self.model_manager:
                return {
                    "insights": ["Pattern interpretation not available"],
                    "recommendations": ["Manual analysis required"]
                }
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            if response.success:
                return self._parse_interpretation_response(response.content)
            else:
                return {
                    "insights": ["Pattern interpretation failed"],
                    "recommendations": ["Manual analysis required"]
                }
                
        except Exception as e:
            logger.error(f"Pattern insights generation failed: {e}")
            return {
                "insights": [f"Insight generation error: {str(e)}"],
                "recommendations": ["Manual interpretation required"]
            }
    
    async def _sentiment_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis on text data."""
        try:
            text_data = data.get("text_data", [])
            analysis_granularity = data.get("granularity", "overall")  # overall, sentence, phrase
            
            if not text_data:
                raise ValueError("Text data is required for sentiment analysis")
            
            # Combine text if it's a list
            if isinstance(text_data, list):
                combined_text = " ".join(str(item) for item in text_data)
            else:
                combined_text = str(text_data)
            
            # Build sentiment analysis prompt
            sentiment_prompt = f"""Perform comprehensive sentiment analysis on the following text:

Text to analyze:
{combined_text}

Please provide:
1. Overall sentiment (positive, negative, neutral) with confidence score
2. Emotional tone analysis
3. Key sentiment indicators (words/phrases)
4. Sentiment intensity (weak, moderate, strong)
5. Mixed sentiment detection if applicable
6. Context and nuance considerations

Granularity: {analysis_granularity}

Be objective and evidence-based in your analysis."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=sentiment_prompt,
                temperature=0.2,
                max_tokens=600
            )
            
            if not response.success:
                raise ValueError(f"Sentiment analysis failed: {response.error}")
            
            # Parse sentiment results
            sentiment_results = self._parse_sentiment_response(response.content)
            
            # Add basic metrics
            sentiment_results["text_length"] = len(combined_text)
            sentiment_results["word_count"] = len(combined_text.split())
            sentiment_results["analysis_granularity"] = analysis_granularity
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.SENTIMENT,
                findings=sentiment_results.get("findings", []),
                insights=sentiment_results.get("insights", []),
                recommendations=sentiment_results.get("recommendations", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology="AI-assisted sentiment analysis",
                limitations=["Context dependency", "Subjective interpretation"],
                supporting_data=sentiment_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "sentiment_results": sentiment_results,
                "text_length": len(combined_text),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise
    
    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """Parse sentiment analysis response."""
        results = {
            "overall_sentiment": "neutral",
            "confidence_score": 0.5,
            "emotional_tone": [],
            "sentiment_indicators": [],
            "intensity": "moderate",
            "findings": [],
            "insights": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Extract overall sentiment
            if 'overall' in line_lower and ('positive' in line_lower or 'negative' in line_lower or 'neutral' in line_lower):
                if 'positive' in line_lower:
                    results["overall_sentiment"] = "positive"
                elif 'negative' in line_lower:
                    results["overall_sentiment"] = "negative"
                else:
                    results["overall_sentiment"] = "neutral"
                
                # Extract confidence if present
                import re
                conf_match = re.search(r'confidence[:\s]*([0-9.]+)', line_lower)
                if conf_match:
                    try:
                        conf_val = float(conf_match.group(1))
                        results["confidence_score"] = conf_val if conf_val <= 1.0 else conf_val / 100.0
                    except:
                        pass
            
            # Extract intensity
            if 'intensity' in line_lower:
                if 'strong' in line_lower:
                    results["intensity"] = "strong"
                elif 'weak' in line_lower:
                    results["intensity"] = "weak"
                else:
                    results["intensity"] = "moderate"
            
            # Identify sections
            if 'tone' in line_lower or 'emotional' in line_lower:
                current_section = 'emotional_tone'
            elif 'indicators' in line_lower or 'keywords' in line_lower:
                current_section = 'sentiment_indicators'
            elif 'findings' in line_lower:
                current_section = 'findings'
            elif 'insights' in line_lower:
                current_section = 'insights'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    results[current_section].append(item)
        
        return results
    
    async def _risk_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk analysis."""
        try:
            risk_factors = data.get("risk_factors", [])
            impact_scale = data.get("impact_scale", "1-10")
            probability_scale = data.get("probability_scale", "1-10")
            context = data.get("context", "")
            
            if not risk_factors:
                raise ValueError("Risk factors are required for risk analysis")
            
            # Build risk analysis prompt
            risk_prompt = f"""Perform comprehensive risk analysis:

Context: {context}

Risk Factors to Analyze:
{chr(10).join([f"- {rf}" for rf in risk_factors])}

Impact Scale: {impact_scale}
Probability Scale: {probability_scale}

Please provide:
1. Risk assessment matrix with impact and probability scores
2. Risk prioritization (high, medium, low)
3. Potential consequences for each risk
4. Mitigation strategies for high-priority risks
5. Contingency planning recommendations
6. Overall risk profile assessment
7. Monitoring and early warning indicators

Use structured risk management methodology."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=risk_prompt,
                temperature=0.2,
                max_tokens=1200
            )
            
            if not response.success:
                raise ValueError(f"Risk analysis failed: {response.error}")
            
            # Parse risk results
            risk_results = self._parse_risk_response(response.content, risk_factors)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.RISK,
                findings=risk_results.get("findings", []),
                insights=risk_results.get("insights", []),
                recommendations=risk_results.get("mitigation_strategies", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology="Qualitative risk analysis",
                limitations=["Subjective scoring", "Qualitative assessment"],
                supporting_data=risk_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "risk_factors": risk_factors,
                "context": context,
                "risk_results": risk_results,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            raise
    
    def _parse_risk_response(self, response: str, risk_factors: List[str]) -> Dict[str, Any]:
        """Parse risk analysis response."""
        results = {
            "risk_matrix": {},
            "high_priority_risks": [],
            "mitigation_strategies": [],
            "contingency_plans": [],
            "findings": [],
            "insights": [],
            "overall_risk_level": "medium"
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'matrix' in line_lower or 'assessment' in line_lower:
                current_section = 'risk_matrix'
            elif 'high priority' in line_lower or 'critical' in line_lower:
                current_section = 'high_priority_risks'
            elif 'mitigation' in line_lower or 'strategies' in line_lower:
                current_section = 'mitigation_strategies'
            elif 'contingency' in line_lower or 'backup' in line_lower:
                current_section = 'contingency_plans'
            elif 'findings' in line_lower:
                current_section = 'findings'
            elif 'insights' in line_lower:
                current_section = 'insights'
            elif 'overall' in line_lower and 'risk' in line_lower:
                if 'high' in line_lower:
                    results["overall_risk_level"] = "high"
                elif 'low' in line_lower:
                    results["overall_risk_level"] = "low"
                else:
                    results["overall_risk_level"] = "medium"
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    if isinstance(results[current_section], list):
                        results[current_section].append(item)
                    elif isinstance(results[current_section], dict):
                        results[current_section][f"item_{len(results[current_section])}"] = item
        
        return results
    
    async def _causal_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform causal analysis."""
        try:
            variables = data.get("variables", [])
            outcome_variable = data.get("outcome_variable", "")
            potential_causes = data.get("potential_causes", [])
            context = data.get("context", "")
            
            if not outcome_variable:
                raise ValueError("Outcome variable is required for causal analysis")
            
            if not potential_causes:
                raise ValueError("Potential causes are required for causal analysis")
            
            # Build causal analysis prompt
            causal_prompt = f"""Perform causal analysis to understand relationships:

Context: {context}

Outcome Variable: {outcome_variable}

Potential Causes:
{chr(10).join([f"- {cause}" for cause in potential_causes])}

Variables Available:
{chr(10).join([f"- {var}" for var in variables])}

Please provide:
1. Causal hypothesis for each potential cause
2. Evidence supporting or refuting each causal relationship
3. Confounding variables to consider
4. Temporal relationships and causation direction
5. Strength of causal evidence (strong, moderate, weak)
6. Alternative explanations
7. Recommendations for establishing causation

Use rigorous causal inference methodology."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=causal_prompt,
                temperature=0.2,
                max_tokens=1200
            )
            
            if not response.success:
                raise ValueError(f"Causal analysis failed: {response.error}")
            
            # Parse causal results
            causal_results = self._parse_causal_response(response.content, potential_causes)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.CAUSAL,
                findings=causal_results.get("findings", []),
                insights=causal_results.get("insights", []),
                recommendations=causal_results.get("recommendations", []),
                confidence_level=ConfidenceLevel.MEDIUM,
                methodology="Qualitative causal analysis",
                limitations=["Observational data", "Confounding variables", "Causal inference challenges"],
                supporting_data=causal_results
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "outcome_variable": outcome_variable,
                "potential_causes": potential_causes,
                "causal_results": causal_results,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            raise
    
    def _parse_causal_response(self, response: str, potential_causes: List[str]) -> Dict[str, Any]:
        """Parse causal analysis response."""
        results = {
            "causal_hypotheses": {},
            "evidence_strength": {},
            "confounding_variables": [],
            "alternative_explanations": [],
            "findings": [],
            "insights": [],
            "recommendations": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'hypothesis' in line_lower or 'hypotheses' in line_lower:
                current_section = 'causal_hypotheses'
            elif 'evidence' in line_lower or 'strength' in line_lower:
                current_section = 'evidence_strength'
            elif 'confounding' in line_lower or 'confounders' in line_lower:
                current_section = 'confounding_variables'
            elif 'alternative' in line_lower or 'explanations' in line_lower:
                current_section = 'alternative_explanations'
            elif 'findings' in line_lower:
                current_section = 'findings'
            elif 'insights' in line_lower:
                current_section = 'insights'
            elif 'recommendations' in line_lower:
                current_section = 'recommendations'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    if isinstance(results[current_section], list):
                        results[current_section].append(item)
                    elif isinstance(results[current_section], dict):
                        results[current_section][f"item_{len(results[current_section])}"] = item
        
        return results
    
    async def _predictive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform predictive analysis."""
        try:
            historical_data = data.get("historical_data", [])
            target_variable = data.get("target_variable", "")
            prediction_horizon = data.get("prediction_horizon", "short_term")
            confidence_interval = data.get("confidence_interval", 0.95)
            
            if not historical_data:
                raise ValueError("Historical data is required for predictive analysis")
            
            # Simple trend-based prediction
            if isinstance(historical_data, list) and all(isinstance(x, (int, float)) for x in historical_data):
                predictions = await self._simple_trend_prediction(historical_data, prediction_horizon)
            else:
                # Use AI for complex predictions
                predictions = await self._ai_assisted_prediction(historical_data, target_variable, prediction_horizon)
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.PREDICTIVE,
                findings=predictions.get("findings", []),
                insights=predictions.get("insights", []),
                recommendations=predictions.get("recommendations", []),
                confidence_level=ConfidenceLevel.LOW,
                methodology=f"Predictive analysis ({prediction_horizon})",
                limitations=["Limited historical data", "Assumption of trend continuation", "External factors not considered"],
                supporting_data=predictions
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "target_variable": target_variable,
                "prediction_horizon": prediction_horizon,
                "predictions": predictions,
                "confidence_interval": confidence_interval,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Predictive analysis failed: {e}")
            raise
    
    async def _simple_trend_prediction(self, data: List[float], horizon: str) -> Dict[str, Any]:
        """Simple trend-based prediction for numerical data."""
        try:
            if len(data) < 3:
                return {"error": "Insufficient data for prediction"}
            
            # Calculate trend
            n = len(data)
            x_values = list(range(n))
            
            # Simple linear regression for trend
            sum_x = sum(x_values)
            sum_y = sum(data)
            sum_xy = sum(x * y for x, y in zip(x_values, data))
            sum_x_sq = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Determine prediction steps
            horizon_steps = {
                "short_term": min(3, max(1, n // 4)),
                "medium_term": min(6, max(2, n // 2)),
                "long_term": min(12, max(3, n))
            }
            
            steps = horizon_steps.get(horizon, 3)
            
            # Generate predictions
            predictions = []
            for i in range(1, steps + 1):
                predicted_value = slope * (n + i - 1) + intercept
                predictions.append({
                    "step": i,
                    "predicted_value": predicted_value,
                    "confidence": max(0.3, 0.8 - (i * 0.1))  # Decreasing confidence
                })
            
            # Calculate prediction uncertainty
            residuals = [data[i] - (slope * i + intercept) for i in range(n)]
            rmse = math.sqrt(sum(r * r for r in residuals) / n)
            
            results = {
                "predictions": predictions,
                "trend_slope": slope,
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "uncertainty": rmse,
                "prediction_horizon": horizon,
                "findings": [],
                "insights": [],
                "recommendations": []
            }
            
            # Generate findings
            if abs(slope) > rmse:
                results["findings"].append(f"Strong {results['trend_direction']} trend detected")
            else:
                results["findings"].append("Weak or unstable trend detected")
            
            if rmse > statistics.mean(data) * 0.2:
                results["findings"].append("High prediction uncertainty due to data volatility")
            
            results["insights"].append(f"Predicted {len(predictions)} future values with decreasing confidence")
            results["recommendations"].append("Monitor actual values against predictions to validate trend")
            
            return results
            
        except Exception as e:
            logger.error(f"Simple trend prediction failed: {e}")
            return {"error": str(e)}
    
    async def _ai_assisted_prediction(self, historical_data: List[Any], target_variable: str, horizon: str) -> Dict[str, Any]:
        """AI-assisted prediction for complex data."""
        try:
            prediction_prompt = f"""Perform predictive analysis based on the following historical data:

Historical Data:
{json.dumps(historical_data[-20:], indent=2, default=str)}  # Last 20 data points

Target Variable: {target_variable}
Prediction Horizon: {horizon}

Please provide:
1. Analysis of historical patterns and trends
2. Key factors influencing the target variable
3. Predictions for the specified horizon
4. Confidence levels for predictions
5. Potential scenarios (optimistic, realistic, pessimistic)
6. Assumptions underlying the predictions
7. Risk factors that could affect predictions

Base your analysis on observable patterns in the data."""
            
            if not self.model_manager:
                return {
                    "findings": ["AI-assisted prediction not available"],
                    "recommendations": ["Use statistical methods or manual analysis"]
                }
            
            response = await self.model_manager.generate_completion(
                prompt=prediction_prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            if response.success:
                return self._parse_prediction_response(response.content)
            else:
                return {
                    "findings": ["Prediction generation failed"],
                    "recommendations": ["Manual analysis required"]
                }
                
        except Exception as e:
            logger.error(f"AI-assisted prediction failed: {e}")
            return {
                "findings": [f"Prediction error: {str(e)}"],
                "recommendations": ["Manual analysis required"]
            }
    
    def _parse_prediction_response(self, response: str) -> Dict[str, Any]:
        """Parse AI prediction response."""
        results = {
            "predictions": [],
            "scenarios": {},
            "key_factors": [],
            "assumptions": [],
            "risk_factors": [],
            "findings": [],
            "insights": [],
            "recommendations": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            if 'predictions' in line_lower or 'forecast' in line_lower:
                current_section = 'predictions'
            elif 'scenarios' in line_lower:
                current_section = 'scenarios'
            elif 'factors' in line_lower or 'influences' in line_lower:
                current_section = 'key_factors'
            elif 'assumptions' in line_lower:
                current_section = 'assumptions'
            elif 'risk' in line_lower:
                current_section = 'risk_factors'
            elif 'findings' in line_lower:
                current_section = 'findings'
            elif 'insights' in line_lower:
                current_section = 'insights'
            elif 'recommendations' in line_lower:
                current_section = 'recommendations'
            elif line.startswith(('- ', '* ', '1. ', '2. ')) and current_section:
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if item and current_section in results:
                    if isinstance(results[current_section], list):
                        results[current_section].append(item)
                    elif isinstance(results[current_section], dict):
                        results[current_section][f"item_{len(results[current_section])}"] = item
        
        return results
    
    async def _data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        try:
            dataset = data.get("dataset", [])
            summary_type = data.get("summary_type", "comprehensive")
            
            if not dataset:
                raise ValueError("Dataset is required for data summary")
            
            # Validate dataset
            validated_data = await self._validate_dataset(dataset)
            summary = validated_data["summary"]
            
            # Generate comprehensive summary
            data_summary_result = {
                "dataset_overview": {
                    "total_records": summary.total_records,
                    "total_columns": len(summary.data_types),
                    "data_quality_score": summary.data_quality_score,
                    "anomalies_detected": summary.anomalies_detected
                },
                "data_types": {k: v.value for k, v in summary.data_types.items()},
                "missing_values": summary.missing_values,
                "summary_statistics": summary.summary_stats,
                "data_quality_assessment": await self._assess_data_quality(summary),
                "findings": [],
                "recommendations": []
            }
            
            # Generate findings
            if summary.data_quality_score < 0.8:
                data_summary_result["findings"].append("Data quality concerns detected")
                data_summary_result["recommendations"].append("Address missing values and data inconsistencies")
            
            if summary.anomalies_detected > 0:
                data_summary_result["findings"].append(f"{summary.anomalies_detected} anomalies detected")
                data_summary_result["recommendations"].append("Investigate anomalies for data validation")
            
            # Identify potential analysis opportunities
            numerical_cols = [col for col, dtype in summary.data_types.items() if dtype == DataType.NUMERICAL]
            if len(numerical_cols) >= 2:
                data_summary_result["recommendations"].append("Consider correlation analysis between numerical variables")
            
            if summary.total_records >= 100:
                data_summary_result["recommendations"].append("Dataset size suitable for statistical analysis")
            
            analysis_result = AnalysisResult(
                analysis_type=AnalysisType.STATISTICAL,
                findings=data_summary_result["findings"],
                insights=data_summary_result["recommendations"],
                recommendations=data_summary_result["recommendations"],
                confidence_level=ConfidenceLevel.HIGH,
                methodology="Comprehensive data profiling",
                limitations=["Basic statistical summary"],
                supporting_data=data_summary_result
            )
            
            # Store analysis
            analysis_id = await self._store_analysis(analysis_result, data)
            
            result = {
                "analysis_id": analysis_id,
                "summary_type": summary_type,
                "data_summary": data_summary_result,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Data summary failed: {e}")
            raise
    
    async def _assess_data_quality(self, summary: DataSummary) -> Dict[str, Any]:
        """Assess data quality based on summary statistics."""
        quality_assessment = {
            "completeness": summary.data_quality_score,
            "consistency": 1.0,  # Would require more analysis
            "accuracy": 1.0,     # Would require validation against known values
            "timeliness": 1.0,   # Would require timestamp analysis
            "overall_score": summary.data_quality_score,
            "issues": [],
            "recommendations": []
        }
        
        # Identify quality issues
        if summary.data_quality_score < 0.9:
            quality_assessment["issues"].append("Missing values detected")
            quality_assessment["recommendations"].append("Implement data cleaning procedures")
        
        if summary.anomalies_detected > summary.total_records * 0.05:
            quality_assessment["issues"].append("High number of anomalies")
            quality_assessment["recommendations"].append("Investigate data collection process")
        
        return quality_assessment
    
    async def _store_analysis(self, analysis_result: AnalysisResult, original_data: Dict[str, Any]) -> str:
        """Store analysis result and return analysis ID."""
        try:
            analysis_id = hashlib.md5(f"{analysis_result.analysis_type.value}_{datetime.now()}".encode()).hexdigest()[:12]
            
            self.active_analyses[analysis_id] = {
                "analysis_result": analysis_result,
                "original_data": original_data,
                "created_at": datetime.now()
            }
            
            # Store in memory
            await self.store_memory(
                content={
                    "analysis_id": analysis_id,
                    "analysis_type": analysis_result.analysis_type.value,
                    "findings": analysis_result.findings,
                    "insights": analysis_result.insights,
                    "recommendations": analysis_result.recommendations,
                    "confidence_level": analysis_result.confidence_level.value,
                    "methodology": analysis_result.methodology,
                    "timestamp": analysis_result.timestamp.isoformat()
                },
                memory_type="analysis_result",
                importance=3.0,
                tags=["analysis", analysis_result.analysis_type.value, analysis_result.confidence_level.value]
            )
            
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")
            return ""
    
    async def _agent_maintenance(self):
        """Analysis agent specific maintenance tasks."""
        try:
            # Update analysis statistics
            await self._update_analysis_statistics()
            
            # Clean up old analyses
            await self._cleanup_old_analyses()
            
            # Optimize reasoning frameworks
            await self._optimize_reasoning_frameworks()
            
        except Exception as e:
            logger.error(f"Analysis agent maintenance failed: {e}")
    
    async def _update_analysis_statistics(self):
        """Update analysis performance statistics."""
        try:
            # Calculate success rate
            total_analyses = self.analysis_stats["total_analyses"]
            successful_analyses = self.analysis_stats["successful_analyses"]
            
            if total_analyses > 0:
                success_rate = successful_analyses / total_analyses
                self.analysis_stats["success_rate"] = success_rate
            
            # Update confidence distribution from recent analyses
            recent_analyses = await self.retrieve_memory(
                memory_type="analysis_result",
                limit=50
            )
            
            confidence_counts = {level.value: 0 for level in ConfidenceLevel}
            for analysis in recent_analyses:
                conf_level = analysis.content.get("confidence_level", "medium")
                if conf_level in confidence_counts:
                    confidence_counts[conf_level] += 1
            
            self.analysis_stats["confidence_distribution"] = confidence_counts
            
            logger.debug(f"Updated analysis statistics: {success_rate:.2%} success rate")
            
        except Exception as e:
            logger.error(f"Analysis statistics update failed: {e}")
    
    async def _cleanup_old_analyses(self):
        """Clean up old analysis results."""
        try:
            # Remove analyses older than 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            analyses_to_remove = []
            for analysis_id, analysis_data in self.active_analyses.items():
                if analysis_data["created_at"] < cutoff_date:
                    analyses_to_remove.append(analysis_id)
            
            # Remove old analyses
            for analysis_id in analyses_to_remove[:20]:  # Limit removal
                if analysis_id in self.active_analyses:
                    del self.active_analyses[analysis_id]
            
            if analyses_to_remove:
                logger.debug(f"Cleaned up {len(analyses_to_remove)} old analyses")
                
        except Exception as e:
            logger.error(f"Analysis cleanup failed: {e}")
    
    async def _optimize_reasoning_frameworks(self):
        """Optimize reasoning frameworks based on usage patterns."""
        try:
            # Analyze which reasoning types are most successful
            recent_reasoning = await self.retrieve_memory(
                memory_type="analysis_result",
                tags=["logical"],
                limit=20
            )
            
            if not recent_reasoning:
                return
            
            # Track successful reasoning patterns for future optimization
            reasoning_patterns = {}
            for analysis in recent_reasoning:
                confidence = analysis.content.get("confidence_level", "medium")
                methodology = analysis.content.get("methodology", "unknown")
                
                if methodology not in reasoning_patterns:
                    reasoning_patterns[methodology] = {"count": 0, "high_confidence": 0}
                
                reasoning_patterns[methodology]["count"] += 1
                if confidence in ["high", "very_high"]:
                    reasoning_patterns[methodology]["high_confidence"] += 1
            
            # Store patterns for future reference
            await self.store_memory(
                content=reasoning_patterns,
                memory_type="reasoning_patterns",
                importance=2.0,
                tags=["reasoning", "optimization", "patterns"]
            )
            
        except Exception as e:
            logger.error(f"Reasoning framework optimization failed: {e}")
    
    async def _agent_shutdown(self):
        """Analysis agent specific shutdown tasks."""
        try:
            # Save analysis statistics
            final_stats = {
                **self.analysis_stats,
                "active_analyses_count": len(self.active_analyses),
                "shutdown_time": datetime.now().isoformat(),
                "agent_runtime": (datetime.now() - self.created_at).total_seconds()
            }
            
            await self.store_memory(
                content=final_stats,
                memory_type="analysis_statistics",
                importance=3.0,
                tags=["statistics", "performance", "shutdown"]
            )
            
            # Save reasoning frameworks backup
            if self.reasoning_frameworks:
                await self.store_memory(
                    content=self.reasoning_frameworks,
                    memory_type="reasoning_backup",
                    importance=3.0,
                    tags=["reasoning", "backup", "shutdown"]
                )
            
            logger.info("Analysis agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Analysis agent shutdown error: {e}")
    
    # Additional utility methods
    
    def get_analysis_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific analysis."""
        if analysis_id not in self.active_analyses:
            return None
        
        analysis_data = self.active_analyses[analysis_id]
        analysis_result = analysis_data["analysis_result"]
        
        return {
            "analysis_id": analysis_id,
            "analysis_type": analysis_result.analysis_type.value,
            "confidence_level": analysis_result.confidence_level.value,
            "methodology": analysis_result.methodology,
            "findings_count": len(analysis_result.findings),
            "insights_count": len(analysis_result.insights),
            "recommendations_count": len(analysis_result.recommendations),
            "created_at": analysis_data["created_at"].isoformat(),
            "timestamp": analysis_result.timestamp.isoformat()
        }
    
    def get_active_analyses(self) -> List[Dict[str, Any]]:
        """Get list of all active analyses."""
        return [
            self.get_analysis_summary(analysis_id)
            for analysis_id in self.active_analyses.keys()
        ]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        return {
            **self.analysis_stats,
            "active_analyses": len(self.active_analyses),
            "reasoning_frameworks": len(self.reasoning_frameworks),
            "problem_solvers": len(self.problem_solvers),
            "agent_uptime": (datetime.now() - self.created_at).total_seconds(),
            "last_updated": datetime.now().isoformat()
        }
    
    async def search_analyses(self, query: str, analysis_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search analyses by content or type."""
        try:
            # Search memory for relevant analyses
            search_tags = [query.replace(" ", "_")]
            if analysis_types:
                search_tags.extend(analysis_types)
            
            relevant_analyses = []
            
            memories = await self.retrieve_memory(
                memory_type="analysis_result",
                tags=search_tags,
                limit=20
            )
            
            for memory in memories:
                content_str = str(memory.content).lower()
                query_words = set(query.lower().split())
                content_words = set(content_str.split())
                
                # Calculate relevance
                overlap = len(query_words.intersection(content_words))
                relevance = overlap / len(query_words) if query_words else 0
                
                if relevance > 0.1:
                    relevant_analyses.append({
                        "memory_id": memory.memory_id,
                        "analysis_type": memory.content.get("analysis_type", "unknown"),
                        "relevance_score": relevance,
                        "confidence_level": memory.content.get("confidence_level", "medium"),
                        "findings_preview": memory.content.get("findings", [])[:3],
                        "created_at": memory.created_at.isoformat()
                    })
            
            # Sort by relevance
            relevant_analyses.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return relevant_analyses[:10]
            
        except Exception as e:
            logger.error(f"Analysis search failed: {e}")
            return []
    
    async def export_analysis_data(self, format: str = "json") -> Dict[str, Any]:
        """Export analysis data in specified format."""
        try:
            export_data = {
                "agent_info": {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "created_at": self.created_at.isoformat(),
                    "capabilities": list(self.capabilities),
                    "analysis_depth": self.analysis_depth,
                    "confidence_threshold": self.confidence_threshold
                },
                "analysis_statistics": self.analysis_stats,
                "active_analyses": {
                    analysis_id: {
                        "analysis_type": data["analysis_result"].analysis_type.value,
                        "confidence_level": data["analysis_result"].confidence_level.value,
                        "methodology": data["analysis_result"].methodology,
                        "findings_count": len(data["analysis_result"].findings),
                        "created_at": data["created_at"].isoformat()
                    }
                    for analysis_id, data in list(self.active_analyses.items())[:20]  # Limit for export
                },
                "reasoning_frameworks": dict(list(self.reasoning_frameworks.items())[:10]),
                "export_info": {
                    "format": format,
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            if format == "json":
                import json
                return {
                    "success": True,
                    "data": json.loads(json.dumps(export_data, default=str)),
                    "format": format
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported export format: {format}",
                    "supported_formats": ["json"]
                }
                
        except Exception as e:
            logger.error(f"Analysis data export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
