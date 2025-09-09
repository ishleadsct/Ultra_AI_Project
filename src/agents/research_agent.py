"""
Ultra AI Project - Research Agent

Specialized agent for information gathering, research, web search,
document analysis, and knowledge synthesis tasks.
"""

import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import hashlib

from .base_agent import BaseAgent, AgentConfig, Task, TaskStatus, AgentCapability
from ..utils.logger import get_logger
from ..utils.helpers import sanitize_string, current_timestamp

logger = get_logger(__name__)

class SearchResult:
    """Search result structure."""
    
    def __init__(self, title: str, url: str, snippet: str, source: str = "web",
                 relevance_score: float = 0.0, timestamp: datetime = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.relevance_score = relevance_score
        self.timestamp = timestamp or datetime.now()
        self.content = ""
        self.metadata = {}

class ResearchReport:
    """Research report structure."""
    
    def __init__(self, query: str, topic: str = ""):
        self.query = query
        self.topic = topic or query
        self.sources = []
        self.key_findings = []
        self.summary = ""
        self.conclusions = []
        self.methodology = ""
        self.limitations = []
        self.created_at = datetime.now()
        self.confidence_score = 0.0

class KnowledgeGraph:
    """Knowledge graph for organizing research findings."""
    
    def __init__(self):
        self.entities = {}  # entity_id: {name, type, attributes}
        self.relationships = {}  # relation_id: {from, to, type, weight}
        self.concepts = {}  # concept_id: {name, definition, examples}
    
    def add_entity(self, name: str, entity_type: str, attributes: Dict[str, Any] = None) -> str:
        """Add an entity to the knowledge graph."""
        entity_id = hashlib.md5(f"{name}_{entity_type}".encode()).hexdigest()[:12]
        self.entities[entity_id] = {
            "name": name,
            "type": entity_type,
            "attributes": attributes or {},
            "created_at": datetime.now()
        }
        return entity_id
    
    def add_relationship(self, from_entity: str, to_entity: str, 
                        relation_type: str, weight: float = 1.0) -> str:
        """Add a relationship between entities."""
        relation_id = hashlib.md5(f"{from_entity}_{to_entity}_{relation_type}".encode()).hexdigest()[:12]
        self.relationships[relation_id] = {
            "from": from_entity,
            "to": to_entity,
            "type": relation_type,
            "weight": weight,
            "created_at": datetime.now()
        }
        return relation_id

class ResearchAgent(BaseAgent):
    """Specialized agent for research and information gathering tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Set up research agent configuration
        if config is None:
            config = {}
        
        agent_config = AgentConfig(
            name=config.get("name", "research_agent"),
            agent_type="research",
            max_concurrent_tasks=config.get("max_concurrent_tasks", 5),
            timeout=config.get("timeout", 1200.0),  # 20 minutes for research tasks
            memory_limit=config.get("memory_limit", 2000),
            capabilities=[
                AgentCapability.RESEARCH.value,
                AgentCapability.WEB_SEARCH.value,
                AgentCapability.DOCUMENT_ANALYSIS.value,
                AgentCapability.SUMMARIZATION.value,
                AgentCapability.TEXT_GENERATION.value
            ],
            preferred_models=config.get("preferred_models", ["gpt-4", "claude-3-opus"]),
            enable_memory=config.get("enable_memory", True),
            custom_settings=config.get("custom_settings", {})
        )
        
        super().__init__(agent_config, **kwargs)
        
        # Research-specific configuration
        self.max_search_results = config.get("max_search_results", 20)
        self.max_content_length = config.get("max_content_length", 50000)
        self.enable_web_search = config.get("enable_web_search", False)  # Requires external API
        self.search_engines = config.get("search_engines", ["duckduckgo", "google"])
        self.fact_check_sources = config.get("fact_check_sources", [])
        
        # Research state
        self.knowledge_graph = KnowledgeGraph()
        self.research_sessions = {}
        self.source_reliability = {}
        
        # Document processing
        self.supported_formats = {
            "pdf", "doc", "docx", "txt", "html", "md", "json", "csv"
        }
        
        logger.info("ResearchAgent initialized")
    
    async def _agent_initialize(self):
        """Research agent specific initialization."""
        try:
            # Initialize search capabilities
            await self._initialize_search_tools()
            
            # Load research methodology templates
            await self._load_research_templates()
            
            # Initialize fact-checking resources
            await self._initialize_fact_checking()
            
            logger.info("ResearchAgent initialization complete")
            
        except Exception as e:
            logger.error(f"ResearchAgent initialization failed: {e}")
            raise
    
    async def _initialize_search_tools(self):
        """Initialize web search and information retrieval tools."""
        try:
            # Check available search tools
            self.available_search_tools = {}
            
            # Note: In a real implementation, you would initialize actual search APIs
            # For now, we'll simulate search capabilities
            self.available_search_tools = {
                "web_search": self.enable_web_search,
                "document_search": True,
                "knowledge_base": True
            }
            
            logger.debug(f"Available search tools: {self.available_search_tools}")
            
        except Exception as e:
            logger.error(f"Failed to initialize search tools: {e}")
    
    async def _load_research_templates(self):
        """Load research methodology templates."""
        try:
            templates = {
                "systematic_review": {
                    "steps": [
                        "Define research question",
                        "Develop search strategy",
                        "Search multiple databases",
                        "Screen and select sources",
                        "Extract and analyze data",
                        "Synthesize findings",
                        "Draw conclusions"
                    ],
                    "quality_criteria": [
                        "Source credibility",
                        "Methodology rigor",
                        "Sample size adequacy",
                        "Peer review status",
                        "Recency of information"
                    ]
                },
                "exploratory_research": {
                    "steps": [
                        "Broad topic exploration",
                        "Identify key themes",
                        "Deep dive into themes",
                        "Cross-reference sources",
                        "Identify gaps",
                        "Synthesize insights"
                    ],
                    "quality_criteria": [
                        "Source diversity",
                        "Information completeness",
                        "Perspective balance",
                        "Evidence strength"
                    ]
                }
            }
            
            await self.store_memory(
                content=templates,
                memory_type="research_templates",
                importance=3.0,
                tags=["research", "methodology", "templates"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load research templates: {e}")
    
    async def _initialize_fact_checking(self):
        """Initialize fact-checking capabilities."""
        try:
            # Store fact-checking guidelines
            fact_check_guidelines = {
                "source_evaluation": [
                    "Check author credentials",
                    "Verify publication date",
                    "Assess publisher reputation",
                    "Look for peer review",
                    "Check for conflicts of interest"
                ],
                "information_verification": [
                    "Cross-reference multiple sources",
                    "Check primary sources",
                    "Verify statistical claims",
                    "Look for supporting evidence",
                    "Assess methodology quality"
                ],
                "reliability_indicators": {
                    "high": ["peer-reviewed journals", "government agencies", "established institutions"],
                    "medium": ["reputable news sources", "professional organizations", "expert blogs"],
                    "low": ["social media", "unverified websites", "anonymous sources"],
                    "avoid": ["known misinformation sites", "biased sources", "conspiracy theory sites"]
                }
            }
            
            await self.store_memory(
                content=fact_check_guidelines,
                memory_type="fact_checking",
                importance=3.0,
                tags=["fact_checking", "verification", "guidelines"]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize fact-checking: {e}")
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a research-related task."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "web_search":
                return await self._web_search(data)
            elif task_type == "research_query":
                return await self._research_query(data)
            elif task_type == "document_analysis":
                return await self._analyze_document(data)
            elif task_type == "fact_check":
                return await self._fact_check(data)
            elif task_type == "summarize":
                return await self._summarize_content(data)
            elif task_type == "literature_review":
                return await self._literature_review(data)
            elif task_type == "comparative_analysis":
                return await self._comparative_analysis(data)
            elif task_type == "trend_analysis":
                return await self._trend_analysis(data)
            elif task_type == "expert_consultation":
                return await self._expert_consultation(data)
            elif task_type == "research_report":
                return await self._generate_research_report(data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _web_search(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search for information."""
        try:
            query = data.get("query", "")
            max_results = data.get("max_results", self.max_search_results)
            search_type = data.get("search_type", "general")  # general, academic, news
            filters = data.get("filters", {})
            
            if not query:
                raise ValueError("Search query is required")
            
            # For now, simulate search results since we don't have actual web search API
            search_results = await self._simulate_web_search(query, max_results, search_type, filters)
            
            # Analyze and rank results
            ranked_results = await self._rank_search_results(search_results, query)
            
            # Store search in memory
            await self.store_memory(
                content={
                    "query": query,
                    "results": [result.__dict__ for result in ranked_results],
                    "search_type": search_type,
                    "timestamp": datetime.now().isoformat()
                },
                memory_type="search_results",
                importance=2.0,
                tags=["web_search", search_type, query.replace(" ", "_")]
            )
            
            return {
                "query": query,
                "results": [result.__dict__ for result in ranked_results],
                "total_results": len(ranked_results),
                "search_type": search_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise
    
    async def _simulate_web_search(self, query: str, max_results: int, 
                                 search_type: str, filters: Dict[str, Any]) -> List[SearchResult]:
        """Simulate web search results (placeholder for actual search API)."""
        # In a real implementation, this would call actual search APIs like:
        # - Google Custom Search API
        # - Bing Search API
        # - DuckDuckGo API
        # - Academic search APIs (Semantic Scholar, etc.)
        
        # For now, return simulated results
        simulated_results = [
            SearchResult(
                title=f"Research on {query} - Academic Paper",
                url=f"https://example.com/paper1",
                snippet=f"This paper presents comprehensive research on {query} with methodology and findings.",
                source="academic",
                relevance_score=0.9
            ),
            SearchResult(
                title=f"{query} - Wikipedia",
                url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                snippet=f"Wikipedia article providing background information on {query}.",
                source="encyclopedia",
                relevance_score=0.8
            ),
            SearchResult(
                title=f"Latest developments in {query}",
                url=f"https://example.com/news1",
                snippet=f"Recent news and developments related to {query}.",
                source="news",
                relevance_score=0.7
            )
        ]
        
        return simulated_results[:max_results]
    
    async def _rank_search_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank search results by relevance and quality."""
        try:
            # Calculate relevance scores based on multiple factors
            for result in results:
                score = 0.0
                
                # Title relevance
                title_words = set(result.title.lower().split())
                query_words = set(query.lower().split())
                title_overlap = len(title_words.intersection(query_words)) / len(query_words)
                score += title_overlap * 0.4
                
                # Snippet relevance
                snippet_words = set(result.snippet.lower().split())
                snippet_overlap = len(snippet_words.intersection(query_words)) / len(query_words)
                score += snippet_overlap * 0.3
                
                # Source quality
                source_quality = {
                    "academic": 0.9,
                    "government": 0.8,
                    "encyclopedia": 0.7,
                    "news": 0.6,
                    "blog": 0.4,
                    "forum": 0.2
                }
                score += source_quality.get(result.source, 0.3) * 0.3
                
                result.relevance_score = min(score, 1.0)
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to rank search results: {e}")
            return results
    
    async def _research_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive research on a query."""
        try:
            query = data.get("query", "")
            research_type = data.get("research_type", "comprehensive")  # exploratory, systematic, comprehensive
            depth = data.get("depth", "medium")  # shallow, medium, deep
            time_period = data.get("time_period", "recent")  # recent, historical, all
            
            if not query:
                raise ValueError("Research query is required")
            
            # Create research session
            session_id = hashlib.md5(f"{query}_{datetime.now()}".encode()).hexdigest()[:12]
            
            # Step 1: Initial search and information gathering
            search_results = await self._web_search({
                "query": query,
                "max_results": 30,
                "search_type": "general"
            })
            
            # Step 2: Analyze and extract key information
            key_information = await self._extract_key_information(search_results["results"], query)
            
            # Step 3: Build knowledge graph
            await self._build_knowledge_graph(key_information, query)
            
            # Step 4: Generate comprehensive summary
            summary = await self._generate_research_summary(key_information, query, research_type)
            
            # Step 5: Identify gaps and further research needs
            gaps = await self._identify_research_gaps(key_information, query)
            
            research_result = {
                "session_id": session_id,
                "query": query,
                "research_type": research_type,
                "depth": depth,
                "summary": summary,
                "key_findings": key_information.get("findings", []),
                "sources": search_results["results"],
                "knowledge_graph": {
                    "entities": len(self.knowledge_graph.entities),
                    "relationships": len(self.knowledge_graph.relationships)
                },
                "research_gaps": gaps,
                "confidence_level": self._calculate_confidence_level(key_information),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store research session
            self.research_sessions[session_id] = research_result
            
            # Store in memory
            await self.store_memory(
                content=research_result,
                memory_type="research_session",
                importance=3.0,
                tags=["research", research_type, query.replace(" ", "_")]
            )
            
            return research_result
            
        except Exception as e:
            logger.error(f"Research query failed: {e}")
            raise
    
    async def _extract_key_information(self, search_results: List[Dict], query: str) -> Dict[str, Any]:
        """Extract key information from search results."""
        try:
            # Build prompt for information extraction
            results_text = "\n\n".join([
                f"Title: {result['title']}\nSnippet: {result['snippet']}\nSource: {result['source']}"
                for result in search_results[:10]  # Limit to top 10 results
            ])
            
            prompt = f"""Analyze the following search results for the query "{query}" and extract key information:

{results_text}

Please provide:
1. Main findings and facts
2. Key themes and topics
3. Important statistics or data points
4. Expert opinions or quotes
5. Potential controversies or debates
6. Related concepts and terms

Format your response as structured information."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1500
            )
            
            if not response.success:
                raise ValueError(f"Information extraction failed: {response.error}")
            
            # Parse the response into structured format
            extracted_info = self._parse_extraction_response(response.content)
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Key information extraction failed: {e}")
            return {"findings": [], "themes": [], "statistics": [], "debates": []}
    
    def _parse_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured information."""
        # Simple parser - could be enhanced with more sophisticated NLP
        lines = response_text.split('\n')
        
        info = {
            "findings": [],
            "themes": [],
            "statistics": [],
            "opinions": [],
            "debates": [],
            "related_concepts": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify sections
            line_lower = line.lower()
            if 'findings' in line_lower or 'facts' in line_lower:
                current_section = 'findings'
            elif 'themes' in line_lower or 'topics' in line_lower:
                current_section = 'themes'
            elif 'statistics' in line_lower or 'data' in line_lower:
                current_section = 'statistics'
            elif 'opinions' in line_lower or 'quotes' in line_lower:
                current_section = 'opinions'
            elif 'controversies' in line_lower or 'debates' in line_lower:
                current_section = 'debates'
            elif 'related' in line_lower or 'concepts' in line_lower:
                current_section = 'related_concepts'
            elif line.startswith(('- ', '* ', '1. ', '2. ')):
                # List item
                item = re.sub(r'^[-*\d.]\s*', '', line).strip()
                if current_section and item:
                    info[current_section].append(item)
        
        return info
    
    async def _build_knowledge_graph(self, information: Dict[str, Any], query: str):
        """Build knowledge graph from extracted information."""
        try:
            # Add main query as central entity
            query_entity = self.knowledge_graph.add_entity(query, "topic")
            
            # Add themes as entities
            for theme in information.get("themes", []):
                theme_entity = self.knowledge_graph.add_entity(theme, "theme")
                self.knowledge_graph.add_relationship(query_entity, theme_entity, "relates_to")
            
            # Add related concepts
            for concept in information.get("related_concepts", []):
                concept_entity = self.knowledge_graph.add_entity(concept, "concept")
                self.knowledge_graph.add_relationship(query_entity, concept_entity, "related_to")
            
        except Exception as e:
            logger.error(f"Knowledge graph building failed: {e}")
    
    async def _generate_research_summary(self, information: Dict[str, Any], 
                                       query: str, research_type: str) -> str:
        """Generate comprehensive research summary."""
        try:
            findings_text = "\n".join([f"- {finding}" for finding in information.get("findings", [])])
            themes_text = "\n".join([f"- {theme}" for theme in information.get("themes", [])])
            
            prompt = f"""Based on the research conducted on "{query}", create a comprehensive summary:


Key Findings:
{findings_text}

Main Themes:
{themes_text}

Research Type: {research_type}

Please provide:
1. Executive summary (2-3 sentences)
2. Detailed analysis of key points
3. Synthesis of different perspectives
4. Implications and significance
5. Areas needing further research

Write in a clear, academic style appropriate for a research report."""
            
            if not self.model_manager:
                return "Summary generation not available"
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1200
            )
            
            if response.success:
                return response.content
            else:
                return f"Summary generation failed: {response.error}"
                
        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return "Summary generation failed"
    
    async def _identify_research_gaps(self, information: Dict[str, Any], query: str) -> List[str]:
        """Identify gaps in current research."""
        try:
            prompt = f"""Based on the research findings for "{query}", identify potential research gaps and areas that need further investigation:

Findings: {', '.join(information.get('findings', [])[:5])}
Themes: {', '.join(information.get('themes', [])[:5])}

Please identify:
1. Unanswered questions
2. Areas with limited information
3. Conflicting viewpoints that need resolution
4. Emerging areas needing research
5. Methodological gaps

Provide a concise list of research gaps."""
            
            if not self.model_manager:
                return ["Research gap analysis not available"]
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            if response.success:
                # Extract gaps from response
                gaps = []
                lines = response.content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(('- ', '* ', '1. ', '2. ')):
                        gap = re.sub(r'^[-*\d.]\s*', '', line).strip()
                        if gap:
                            gaps.append(gap)
                
                return gaps
            else:
                return ["Research gap analysis failed"]
                
        except Exception as e:
            logger.error(f"Research gap identification failed: {e}")
            return ["Error in gap analysis"]
    
    def _calculate_confidence_level(self, information: Dict[str, Any]) -> float:
        """Calculate confidence level based on available information."""
        try:
            # Simple confidence calculation based on information completeness
            factors = {
                "findings": len(information.get("findings", [])),
                "themes": len(information.get("themes", [])),
                "statistics": len(information.get("statistics", [])),
                "sources": len(information.get("sources", []))
            }
            
            # Normalize scores
            max_possible = {"findings": 10, "themes": 8, "statistics": 5, "sources": 15}
            
            weighted_score = 0.0
            weights = {"findings": 0.3, "themes": 0.2, "statistics": 0.2, "sources": 0.3}
            
            for factor, count in factors.items():
                normalized = min(count / max_possible.get(factor, 10), 1.0)
                weighted_score += normalized * weights.get(factor, 0.25)
            
            return min(weighted_score, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _analyze_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a document for key information."""
        try:
            document_content = data.get("content", "")
            document_type = data.get("type", "text")
            analysis_focus = data.get("focus", ["summary", "key_points", "themes"])
            
            if not document_content:
                raise ValueError("Document content is required")
            
            # Truncate if too long
            if len(document_content) > self.max_content_length:
                document_content = document_content[:self.max_content_length] + "..."
            
            analysis_results = {}
            
            # Generate summary if requested
            if "summary" in analysis_focus:
                analysis_results["summary"] = await self._generate_document_summary(document_content)
            
            # Extract key points if requested
            if "key_points" in analysis_focus:
                analysis_results["key_points"] = await self._extract_key_points(document_content)
            
            # Identify themes if requested
            if "themes" in analysis_focus:
                analysis_results["themes"] = await self._identify_themes(document_content)
            
            # Extract entities if requested
            if "entities" in analysis_focus:
                analysis_results["entities"] = await self._extract_entities(document_content)
            
            # Sentiment analysis if requested
            if "sentiment" in analysis_focus:
                analysis_results["sentiment"] = await self._analyze_sentiment(document_content)
            
            result = {
                "document_type": document_type,
                "content_length": len(document_content),
                "analysis_focus": analysis_focus,
                "results": analysis_results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store analysis in memory
            await self.store_memory(
                content=result,
                memory_type="document_analysis",
                importance=2.5,
                tags=["document_analysis", document_type] + analysis_focus
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            raise
    
    async def _generate_document_summary(self, content: str) -> str:
        """Generate summary of document content."""
        try:
            prompt = f"""Summarize the following document content:

{content}

Provide a concise but comprehensive summary that captures the main points, key arguments, and important details. Aim for 2-3 paragraphs."""
            
            if not self.model_manager:
                return "Summary generation not available"
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            return response.content if response.success else "Summary generation failed"
            
        except Exception as e:
            logger.error(f"Document summary failed: {e}")
            return "Summary generation failed"
    
    async def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from document content."""
        try:
            prompt = f"""Extract the key points from the following content:

{content}

Provide a bullet-point list of the most important facts, findings, arguments, and conclusions. Focus on actionable insights and significant information."""
            
            if not self.model_manager:
                return ["Key point extraction not available"]
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=600
            )
            
            if response.success:
                # Parse key points from response
                key_points = []
                lines = response.content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(('- ', '* ', '•')):
                        point = line[2:].strip()
                        if point:
                            key_points.append(point)
                
                return key_points
            else:
                return ["Key point extraction failed"]
                
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return ["Error in key point extraction"]
    
    async def _identify_themes(self, content: str) -> List[str]:
        """Identify main themes in document content."""
        try:
            prompt = f"""Identify the main themes and topics discussed in the following content:

{content}

List the primary themes, concepts, and subject areas covered. Focus on recurring topics and central ideas."""
            
            if not self.model_manager:
                return ["Theme identification not available"]
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=400
            )
            
            if response.success:
                # Parse themes from response
                themes = []
                lines = response.content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.endswith(':'):
                        # Remove bullet points and numbering
                        theme = re.sub(r'^[-*\d.]\s*', '', line).strip()
                        if theme and len(theme) > 3:
                            themes.append(theme)
                
                return themes[:10]  # Limit to top 10 themes
            else:
                return ["Theme identification failed"]
                
        except Exception as e:
            logger.error(f"Theme identification failed: {e}")
            return ["Error in theme identification"]
    
    async def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content."""
        try:
            prompt = f"""Extract named entities from the following content:

{content}

Identify and categorize:
- People (names of individuals)
- Organizations (companies, institutions)
- Locations (places, countries, cities)
- Dates (specific dates, time periods)
- Technologies (products, systems, methods)
- Concepts (key terms, methodologies)

Format as categories with lists."""

            if not self.model_manager:
                return {"entities": ["Entity extraction not available"]}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=600
            )
            
            if response.success:
                # Parse entities from response
                entities = {
                    "people": [],
                    "organizations": [],
                    "locations": [],
                    "dates": [],
                    "technologies": [],
                    "concepts": []
                }
                
                lines = response.content.split('\n')
                current_category = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check for category headers
                    line_lower = line.lower()
                    if 'people' in line_lower:
                        current_category = 'people'
                    elif 'organization' in line_lower:
                        current_category = 'organizations'
                    elif 'location' in line_lower:
                        current_category = 'locations'
                    elif 'date' in line_lower:
                        current_category = 'dates'
                    elif 'technolog' in line_lower:
                        current_category = 'technologies'
                    elif 'concept' in line_lower:
                        current_category = 'concepts'
                    elif line.startswith(('- ', '* ')) and current_category:
                        entity = line[2:].strip()
                        if entity:
                            entities[current_category].append(entity)
                
                return entities
            else:
                return {"entities": ["Entity extraction failed"]}
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": ["Error in entity extraction"]}
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of document content."""
        try:
            prompt = f"""Analyze the sentiment and tone of the following content:

{content}

Provide:
1. Overall sentiment (positive, negative, neutral)
2. Confidence score (0-1)
3. Key emotional indicators
4. Tone description (formal, casual, academic, etc.)

Be objective and evidence-based in your analysis."""
            
            if not self.model_manager:
                return {"sentiment": "not_available", "confidence": 0.0}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=300
            )
            
            if response.success:
                # Simple parsing of sentiment response
                content_lower = response.content.lower()
                
                sentiment = "neutral"
                if "positive" in content_lower:
                    sentiment = "positive"
                elif "negative" in content_lower:
                    sentiment = "negative"
                
                # Extract confidence if mentioned
                confidence = 0.7  # Default
                import re
                conf_match = re.search(r'confidence[:\s]*([0-9.]+)', content_lower)
                if conf_match:
                    try:
                        confidence = float(conf_match.group(1))
                        if confidence > 1.0:
                            confidence = confidence / 100.0  # Convert percentage
                    except:
                        pass
                
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "analysis": response.content
                }
            else:
                return {"sentiment": "analysis_failed", "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "error", "confidence": 0.0}
    
    async def _fact_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fact-check information against reliable sources."""
        try:
            claim = data.get("claim", "")
            sources_to_check = data.get("sources", [])
            verification_level = data.get("verification_level", "standard")  # basic, standard, rigorous
            
            if not claim:
                raise ValueError("Claim is required for fact-checking")
            
            # Retrieve fact-checking guidelines
            guidelines = await self.retrieve_memory(memory_type="fact_checking", limit=1)
            
            # Search for verification sources
            search_results = await self._web_search({
                "query": f"fact check verify: {claim}",
                "max_results": 10,
                "search_type": "academic"
            })
            
            # Analyze claim against sources
            verification_result = await self._verify_claim(claim, search_results["results"], verification_level)
            
            result = {
                "claim": claim,
                "verification_level": verification_level,
                "verification_result": verification_result,
                "sources_checked": len(search_results["results"]),
                "confidence_level": verification_result.get("confidence", 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store fact-check in memory
            await self.store_memory(
                content=result,
                memory_type="fact_check",
                importance=3.0,
                tags=["fact_check", "verification", verification_level]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            raise
    
    async def _verify_claim(self, claim: str, sources: List[Dict], verification_level: str) -> Dict[str, Any]:
        """Verify a claim against available sources."""
        try:
            sources_text = "\n\n".join([
                f"Source: {source['title']}\nContent: {source['snippet']}\nURL: {source['url']}"
                for source in sources[:5]  # Limit to top 5 sources
            ])
            
            prompt = f"""Fact-check the following claim against the provided sources:

Claim: "{claim}"

Sources:
{sources_text}

Verification level: {verification_level}

Please provide:
1. Verification status (TRUE, FALSE, PARTIALLY_TRUE, UNVERIFIED)
2. Confidence level (0-1)
3. Supporting evidence
4. Contradicting evidence
5. Source reliability assessment
6. Overall assessment

Be thorough and objective in your analysis."""
            
            if not self.model_manager:
                return {"status": "UNVERIFIED", "confidence": 0.0, "reason": "Verification not available"}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=800
            )
            
            if response.success:
                # Parse verification result
                content = response.content.lower()
                
                # Determine status
                status = "UNVERIFIED"
                if "true" in content and "false" not in content:
                    status = "TRUE"
                elif "false" in content:
                    if "partially" in content:
                        status = "PARTIALLY_TRUE"
                    else:
                        status = "FALSE"
                
                # Extract confidence
                confidence = 0.5
                import re
                conf_match = re.search(r'confidence[:\s]*([0-9.]+)', content)
                if conf_match:
                    try:
                        confidence = float(conf_match.group(1))
                        if confidence > 1.0:
                            confidence = confidence / 100.0
                    except:
                        pass
                
                return {
                    "status": status,
                    "confidence": confidence,
                    "analysis": response.content,
                    "sources_reliability": "mixed"  # Could be enhanced
                }
            else:
                return {"status": "UNVERIFIED", "confidence": 0.0, "reason": "Verification failed"}
                
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {"status": "ERROR", "confidence": 0.0, "reason": str(e)}
    
    async def _summarize_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize content with specified parameters."""
        try:
            content = data.get("content", "")
            summary_type = data.get("type", "abstractive")  # extractive, abstractive, bullet_points
            length = data.get("length", "medium")  # short, medium, long
            focus = data.get("focus", "general")  # general, key_points, conclusions
            
            if not content:
                raise ValueError("Content is required for summarization")
            
            # Determine summary length
            max_tokens = {
                "short": 150,
                "medium": 300,
                "long": 600
            }.get(length, 300)
            
            # Build summarization prompt
            prompt = f"""Summarize the following content:

{content}

Summary type: {summary_type}
Length: {length}
Focus: {focus}

Please provide a {length} {summary_type} summary focusing on {focus}."""
            
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=max_tokens
            )
            
            if not response.success:
                raise ValueError(f"Summarization failed: {response.error}")
            
            result = {
                "original_length": len(content),
                "summary": response.content,
                "summary_type": summary_type,
                "length": length,
                "focus": focus,
                "compression_ratio": len(response.content) / len(content),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store summary in memory
            await self.store_memory(
                content=result,
                memory_type="summary",
                importance=2.0,
                tags=["summarization", summary_type, length, focus]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Content summarization failed: {e}")
            raise
    
    async def _literature_review(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct a literature review on a topic."""
        try:
            topic = data.get("topic", "")
            scope = data.get("scope", "comprehensive")  # focused, comprehensive, systematic
            time_range = data.get("time_range", "recent")  # recent, historical, all
            
            if not topic:
                raise ValueError("Topic is required for literature review")
            
            # Search for academic sources
            academic_results = await self._web_search({
                "query": f"{topic} academic research papers",
                "max_results": 20,
                "search_type": "academic"
            })
            
            # Analyze and synthesize literature
            synthesis = await self._synthesize_literature(academic_results["results"], topic, scope)
            
            # Identify trends and gaps
            trends = await self._identify_research_trends(academic_results["results"], topic)
            gaps = await self._identify_research_gaps(synthesis, topic)
            
            result = {
                "topic": topic,
                "scope": scope,
                "time_range": time_range,
                "sources_reviewed": len(academic_results["results"]),
                "synthesis": synthesis,
                "research_trends": trends,
                "research_gaps": gaps,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store literature review
            await self.store_memory(
                content=result,
                memory_type="literature_review",
                importance=3.5,
                tags=["literature_review", scope, topic.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Literature review failed: {e}")
            raise
    
    async def _synthesize_literature(self, sources: List[Dict], topic: str, scope: str) -> Dict[str, Any]:
        """Synthesize literature findings."""
        try:
            sources_text = "\n\n".join([
                f"Title: {source['title']}\nSnippet: {source['snippet']}"
                for source in sources[:10]
            ])
            
            prompt = f"""Synthesize the following literature on "{topic}":

{sources_text}

Scope: {scope}

Please provide:
1. Overview of current state of research
2. Key findings and conclusions
3. Methodological approaches used
4. Areas of consensus
5. Areas of disagreement or debate
6. Evolution of thinking over time

Provide a comprehensive synthesis appropriate for a literature review."""
            
            if not self.model_manager:
                return {"synthesis": "Literature synthesis not available"}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1200
            )
            
            if response.success:
                return {"synthesis": response.content, "methodology": "AI-assisted synthesis"}
            else:
                return {"synthesis": "Literature synthesis failed", "error": response.error}
                
        except Exception as e:
            logger.error(f"Literature synthesis failed: {e}")
            return {"synthesis": "Synthesis failed", "error": str(e)}
    
    async def _identify_research_trends(self, sources: List[Dict], topic: str) -> List[str]:
        """Identify research trends from literature."""
        try:
            # This is a simplified implementation
            # In practice, you'd analyze publication dates, citation patterns, etc.
            
            recent_sources = [s for s in sources if "2023" in s.get("snippet", "") or "2024" in s.get("snippet", "")]
            
            if not recent_sources:
                return ["Insufficient recent data for trend analysis"]
            
            trends_text = "\n".join([source["title"] for source in recent_sources])
            
            prompt = f"""Based on these recent research titles related to "{topic}", identify emerging trends:

{trends_text}

List the key research trends, methodological shifts, and emerging directions in this field."""
            
            if not self.model_manager:
                return ["Trend analysis not available"]
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=400
            )
            
            if response.success:
                # Parse trends from response
                trends = []
                lines = response.content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(('- ', '* ', '1. ')):
                        trend = re.sub(r'^[-*\d.]\s*', '', line).strip()
                        if trend:
                            trends.append(trend)
                
                return trends
            else:
                return ["Trend analysis failed"]
                
        except Exception as e:
            logger.error(f"Research trend identification failed: {e}")
            return ["Error in trend analysis"]
    
    async def _comparative_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis between topics or approaches."""
        try:
            topics = data.get("topics", [])
            comparison_aspects = data.get("aspects", ["similarities", "differences", "advantages", "disadvantages"])
            
            if len(topics) < 2:
                raise ValueError("At least two topics required for comparison")
            
            # Research each topic
            topic_research = {}
            for topic in topics:
                research = await self._research_query({
                    "query": topic,
                    "research_type": "focused",
                    "depth": "medium"
                })
                topic_research[topic] = research
            
            # Perform comparison
            comparison = await self._compare_topics(topic_research, comparison_aspects)
            
            result = {
                "topics": topics,
                "comparison_aspects": comparison_aspects,
                "topic_research": {topic: research["summary"] for topic, research in topic_research.items()},
                "comparison": comparison,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store comparison
            await self.store_memory(
                content=result,
                memory_type="comparative_analysis",
                importance=3.0,
                tags=["comparison", "analysis"] + [topic.replace(" ", "_") for topic in topics]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            raise
    
    async def _compare_topics(self, topic_research: Dict[str, Dict], aspects: List[str]) -> Dict[str, Any]:
        """Compare topics across specified aspects."""
        try:
            topics = list(topic_research.keys())
            summaries = {topic: research["summary"] for topic, research in topic_research.items()}
            
            summaries_text = "\n\n".join([
                f"{topic}:\n{summary}" for topic, summary in summaries.items()
            ])
            
            prompt = f"""Compare the following topics across these aspects: {', '.join(aspects)}

Topics and Information:
{summaries_text}

Provide a detailed comparison covering:
{', '.join(aspects)}

Structure your response clearly with sections for each aspect."""
            
            if not self.model_manager:
                return {"comparison": "Comparison not available"}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1000
            )
            
            if response.success:
                return {"comparison": response.content, "methodology": "AI-assisted comparison"}
            else:
                return {"comparison": "Comparison failed", "error": response.error}
                
        except Exception as e:
            logger.error(f"Topic comparison failed: {e}")
            return {"comparison": "Comparison failed", "error": str(e)}
    
    async def _trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in a specific domain."""
        try:
            domain = data.get("domain", "")
            time_period = data.get("time_period", "5_years")
            trend_types = data.get("trend_types", ["growth", "decline", "emerging", "stable"])
            
            if not domain:
                raise ValueError("Domain is required for trend analysis")
            
            # Search for trend-related information
            trend_results = await self._web_search({
                "query": f"{domain} trends {time_period} analysis",
                "max_results": 15,
                "search_type": "general"
            })
            
            # Analyze trends
            trend_analysis = await self._analyze_trends(trend_results["results"], domain, trend_types)
            
            result = {
                "domain": domain,
                "time_period": time_period,
                "trend_types": trend_types,
                "trend_analysis": trend_analysis,
                "sources_analyzed": len(trend_results["results"]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store trend analysis
            await self.store_memory(
                content=result,
                memory_type="trend_analysis",
                importance=2.5,
                tags=["trends", "analysis", domain.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise
    
    async def _analyze_trends(self, sources: List[Dict], domain: str, trend_types: List[str]) -> Dict[str, Any]:
        """Analyze trends from source materials."""
        try:
            sources_text = "\n\n".join([
                f"Title: {source['title']}\nContent: {source['snippet']}"
                for source in sources[:8]
            ])
            
            prompt = f"""Analyze trends in the "{domain}" domain based on the following sources:

{sources_text}

Focus on these trend types: {', '.join(trend_types)}

Please identify:
1. Current trends (what's happening now)
2. Emerging trends (what's beginning to develop)
3. Declining trends (what's becoming less relevant)
4. Stable patterns (what remains consistent)
5. Future projections (where things are heading)

Provide specific examples and evidence for each trend identified."""
            
            if not self.model_manager:
                return {"trends": "Trend analysis not available"}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            if response.success:
                return {"analysis": response.content, "methodology": "AI-assisted trend analysis"}
            else:
                return {"analysis": "Trend analysis failed", "error": response.error}
                
        except Exception as e:
            logger.error(f"Trend analysis processing failed: {e}")
            return {"analysis": "Analysis failed", "error": str(e)}
    
    async def _expert_consultation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate expert consultation by finding expert opinions."""
        try:
            topic = data.get("topic", "")
            expert_type = data.get("expert_type", "general")
            consultation_type = data.get("consultation_type", "opinion")  # opinion, analysis, recommendation
            
            if not topic:
                raise ValueError("Topic is required for expert consultation")
            
            # Search for expert opinions and analysis
            expert_results = await self._web_search({
                "query": f"{topic} expert opinion analysis {expert_type}",
                "max_results": 10,
                "search_type": "academic"
            })
            
            # Synthesize expert perspectives
            expert_synthesis = await self._synthesize_expert_opinions(
                expert_results["results"], topic, consultation_type
            )
            
            result = {
                "topic": topic,
                "expert_type": expert_type,
                "consultation_type": consultation_type,
                "expert_synthesis": expert_synthesis,
                "sources_consulted": len(expert_results["results"]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store expert consultation
            await self.store_memory(
                content=result,
                memory_type="expert_consultation",
                importance=3.0,
                tags=["expert_consultation", expert_type, topic.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Expert consultation failed: {e}")
            raise
    
    async def _synthesize_expert_opinions(self, sources: List[Dict], topic: str, consultation_type: str) -> Dict[str, Any]:
        """Synthesize expert opinions from sources."""
        try:
            sources_text = "\n\n".join([
                f"Source: {source['title']}\nContent: {source['snippet']}"
                for source in sources[:6]
            ])
            
            prompt = f"""Synthesize expert opinions on "{topic}" from the following sources:

{sources_text}

Consultation type: {consultation_type}

Please provide:
1. Consensus areas (where experts agree)
2. Areas of disagreement or debate
3. Key expert insights and recommendations
4. Different perspectives and their rationales
5. Credibility assessment of sources

Present the synthesis as if consulting multiple experts on this topic."""
            
            if not self.model_manager:
                return {"synthesis": "Expert consultation not available"}
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            if response.success:
                return {"synthesis": response.content, "methodology": "Multi-source expert synthesis"}
            else:
                return {"synthesis": "Expert consultation failed", "error": response.error}
                
        except Exception as e:
            logger.error(f"Expert opinion synthesis failed: {e}")
            return {"synthesis": "Synthesis failed", "error": str(e)}
    
    async def _generate_research_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive research report."""
        try:
            topic = data.get("topic", "")
            report_type = data.get("report_type", "comprehensive")  # brief, standard, comprehensive
            sections = data.get("sections", ["executive_summary", "background", "findings", "analysis", "conclusions", "recommendations"])
            
            if not topic:
                raise ValueError("Topic is required for research report")
            
            # Conduct comprehensive research
            research_data = await self._research_query({
                "query": topic,
                "research_type": "systematic",
                "depth": "deep"
            })
            
            # Generate each section
            report_sections = {}
            for section in sections:
                report_sections[section] = await self._generate_report_section(
                    section, topic, research_data, report_type
                )
            
            # Compile final report
            full_report = await self._compile_research_report(
                topic, report_sections, research_data, report_type
            )
            
            result = {
                "topic": topic,
                "report_type": report_type,
                "sections": sections,
                "report_sections": report_sections,
                "full_report": full_report,
                "research_data": research_data["session_id"],
                "word_count": len(full_report.split()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store research report
            await self.store_memory(
                content=result,
                memory_type="research_report",
                importance=4.0,
                tags=["research_report", report_type, topic.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Research report generation failed: {e}")
            raise
    
    async def _generate_report_section(self, section: str, topic: str, 
                                     research_data: Dict[str, Any], report_type: str) -> str:
        """Generate a specific section of the research report."""
        try:
            section_prompts = {
                "executive_summary": f"Write an executive summary for a research report on '{topic}'. Summarize the key findings, main conclusions, and primary recommendations in 2-3 paragraphs.",
                "background": f"Write a background section for a research report on '{topic}'. Provide context, define key terms, and explain the significance of this topic.",
                "findings": f"Based on the research findings on '{topic}', write a findings section that presents the key discoveries, data, and evidence systematically.",
                "analysis": f"Write an analysis section for a research report on '{topic}'. Interpret the findings, discuss implications, and analyze patterns or relationships.",
                "conclusions": f"Write a conclusions section for a research report on '{topic}'. Synthesize the main insights and present logical conclusions based on the evidence.",
                "recommendations": f"Write a recommendations section for a research report on '{topic}'. Provide actionable recommendations based on the research findings."
            }
            
            prompt = section_prompts.get(section, f"Write a {section} section for a research report on '{topic}'.")
            
            # Add research context
            if research_data.get("summary"):
                prompt += f"\n\nResearch context:\n{research_data['summary'][:500]}..."
            
            if not self.model_manager:
                return f"[{section.title()} section not available]"
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=600 if report_type == "comprehensive" else 300
            )
            
            return response.content if response.success else f"[{section.title()} generation failed]"
            
        except Exception as e:
            logger.error(f"Report section generation failed: {e}")
            return f"[Error generating {section} section]"
    
    async def _compile_research_report(self, topic: str, sections: Dict[str, str], 
                                     research_data: Dict[str, Any], report_type: str) -> str:
        """Compile the final research report."""
        try:
            report_parts = [
                f"# Research Report: {topic}",
                f"",
                f"**Report Type:** {report_type.title()}",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Research Session:** {research_data.get('session_id', 'N/A')}",
                f"",
                "---",
                ""
            ]
            
            # Add each section
            section_titles = {
                "executive_summary": "Executive Summary",
                "background": "Background",
                "findings": "Key Findings",
                "analysis": "Analysis",
                "conclusions": "Conclusions",
                "recommendations": "Recommendations"
            }
            
            for section_key, content in sections.items():
                title = section_titles.get(section_key, section_key.replace("_", " ").title())
                report_parts.extend([
                    f"## {title}",
                    "",
                    content,
                    "",
                    "---",
                    ""
                ])
            
            # Add appendices
            report_parts.extend([
                "## Appendix A: Sources",
                "",
                f"This report is based on {research_data.get('total_results', 'multiple')} sources.",
                f"Research confidence level: {research_data.get('confidence_level', 0.5):.1%}",
                "",
                "## Appendix B: Methodology",
                "",
                "This research report was generated using AI-assisted research methodology,",
                "including web search, document analysis, and synthesis techniques.",
                ""
            ])
            
            return "\n".join(report_parts)
            
        except Exception as e:
            logger.error(f"Report compilation failed: {e}")
            return f"Error compiling research report: {str(e)}"
    
    async def _agent_maintenance(self):
        """Research agent specific maintenance tasks."""
        try:
            # Update source reliability scores
            await self._update_source_reliability()
            
            # Compress knowledge graph if it gets too large
            if len(self.knowledge_graph.entities) > 1000:
                await self._compress_knowledge_graph()
            
            # Clean up old research sessions
            await self._cleanup_research_sessions()
            
        except Exception as e:
            logger.error(f"Research agent maintenance failed: {e}")
    
    async def _update_source_reliability(self):
        """Update reliability scores for sources."""
        try:
            # This would track which sources provide accurate information over time
            # For now, just maintain the structure
            pass
            
        except Exception as e:
            logger.error(f"Source reliability update failed: {e}")
    
    async def _compress_knowledge_graph(self):
        """Compress knowledge graph by removing less important entities."""
        try:
            # Remove entities with low importance or few connections
            entities_to_remove = []
            
            for entity_id, entity in self.knowledge_graph.entities.items():
                # Count connections
                connections = sum(1 for rel in self.knowledge_graph.relationships.values() 
                                if rel["from"] == entity_id or rel["to"] == entity_id)
                
                # Remove if less than 2 connections and old
                if connections < 2:
                    entity_age = (datetime.now() - entity["created_at"]).days
                    if entity_age > 7:  # Older than a week
                        entities_to_remove.append(entity_id)
            
            # Remove entities and their relationships
            for entity_id in entities_to_remove[:100]:  # Limit removal
                if entity_id in self.knowledge_graph.entities:
                    del self.knowledge_graph.entities[entity_id]
                
                # Remove related relationships
                relationships_to_remove = [
                    rel_id for rel_id, rel in self.knowledge_graph.relationships.items()
                    if rel["from"] == entity_id or rel["to"] == entity_id
                ]
                
                for rel_id in relationships_to_remove:
                    del self.knowledge_graph.relationships[rel_id]
            
            if entities_to_remove:
                logger.debug(f"Compressed knowledge graph: removed {len(entities_to_remove)} entities")
                
        except Exception as e:
            logger.error(f"Knowledge graph compression failed: {e}")
    
    async def _cleanup_research_sessions(self):
        """Clean up old research sessions."""
        try:
            # Keep only recent sessions (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            
            sessions_to_remove = []
            for session_id, session_data in self.research_sessions.items():
                session_time = datetime.fromisoformat(session_data.get("timestamp", "1970-01-01T00:00:00"))
                if session_time < cutoff_date:
                    sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove[:50]:  # Limit removal
                if session_id in self.research_sessions:
                    del self.research_sessions[session_id]
            
            if sessions_to_remove:
                logger.debug(f"Cleaned up {len(sessions_to_remove)} old research sessions")
                
        except Exception as e:
            logger.error(f"Research session cleanup failed: {e}")
    
    async def _agent_shutdown(self):
        """Research agent specific shutdown tasks."""
        try:
            # Save knowledge graph to persistent storage (if available)
            knowledge_graph_data = {
                "entities": self.knowledge_graph.entities,
                "relationships": self.knowledge_graph.relationships,
                "concepts": self.knowledge_graph.concepts
            }
            
            await self.store_memory(
                content=knowledge_graph_data,
                memory_type="knowledge_graph_backup",
                importance=4.0,
                tags=["knowledge_graph", "backup", "shutdown"]
            )
            
            # Save research sessions summary
            sessions_summary = {
                "total_sessions": len(self.research_sessions),
                "active_sessions": [sid for sid, data in self.research_sessions.items() 
                                  if (datetime.now() - datetime.fromisoformat(data.get("timestamp", "1970-01-01T00:00:00"))).days < 1],
                "shutdown_time": datetime.now().isoformat()
            }
            
            await self.store_memory(
                content=sessions_summary,
                memory_type="research_summary",
                importance=3.0,
                tags=["research_summary", "shutdown"]
            )
            
            logger.info("Research agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Research agent shutdown error: {e}")
    
    # Additional utility methods
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "entities": len(self.knowledge_graph.entities),
            "relationships": len(self.knowledge_graph.relationships),
            "concepts": len(self.knowledge_graph.concepts),
            "entity_types": list(set(entity["type"] for entity in self.knowledge_graph.entities.values())),
            "relationship_types": list(set(rel["type"] for rel in self.knowledge_graph.relationships.values()))
        }
    
    def get_research_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific research session."""
        if session_id not in self.research_sessions:
            return None
        
        session = self.research_sessions[session_id]
        return {
            "session_id": session_id,
            "query": session.get("query"),
            "research_type": session.get("research_type"),
            "sources_count": len(session.get("sources", [])),
            "findings_count": len(session.get("key_findings", [])),
            "confidence_level": session.get("confidence_level"),
            "timestamp": session.get("timestamp")
        }
    
    def get_active_research_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active research sessions."""
        active_sessions = []
        cutoff_time = datetime.now() - timedelta(hours=24)  # Last 24 hours
        
        for session_id, session_data in self.research_sessions.items():
            session_time = datetime.fromisoformat(session_data.get("timestamp", "1970-01-01T00:00:00"))
            if session_time > cutoff_time:
                active_sessions.append(self.get_research_session_summary(session_id))
        
        return active_sessions
    
    async def search_memory(self, query: str, memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search research memory for relevant information."""
        try:
            # Get relevant memories
            memories = await self.retrieve_memory(
                memory_type=memory_types[0] if memory_types else None,
                tags=[query.replace(" ", "_")],
                limit=20
            )
            
            # Score and rank memories by relevance
            relevant_memories = []
            query_words = set(query.lower().split())
            
            for memory in memories:
                content_str = str(memory.content).lower()
                content_words = set(content_str.split())
                
                # Calculate relevance score
                overlap = len(query_words.intersection(content_words))
                relevance = overlap / len(query_words) if query_words else 0
                
                if relevance > 0.1:  # Minimum relevance threshold
                    relevant_memories.append({
                        "memory_id": memory.memory_id,
                        "memory_type": memory.memory_type,
                        "relevance_score": relevance,
                        "importance": memory.importance,
                        "content_preview": content_str[:200] + "..." if len(content_str) > 200 else content_str,
                        "tags": memory.tags,
                        "created_at": memory.created_at.isoformat(),
                        "accessed_at": memory.accessed_at.isoformat()
                    })
            
            # Sort by relevance and importance
            relevant_memories.sort(key=lambda x: (x["relevance_score"], x["importance"]), reverse=True)
            
            return relevant_memories[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def export_research_data(self, format: str = "json") -> Dict[str, Any]:
        """Export research data in specified format."""
        try:
            export_data = {
                "agent_info": {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "created_at": self.created_at.isoformat(),
                    "capabilities": list(self.capabilities)
                },
                "knowledge_graph": {
                    "entities": dict(list(self.knowledge_graph.entities.items())[:100]),  # Limit for export
                    "relationships": dict(list(self.knowledge_graph.relationships.items())[:100]),
                    "concepts": dict(list(self.knowledge_graph.concepts.items())[:50])
                },
                "research_sessions": {
                    session_id: {
                        k: v for k, v in session_data.items() 
                        if k not in ["sources"]  # Exclude large source data
                    }
                    for session_id, session_data in list(self.research_sessions.items())[:20]
                },
                "memory_summary": {
                    "total_memories": len(self.agent_memory),
                    "memory_types": list(set(memory.memory_type for memory in self.agent_memory.values())),
                    "recent_memories": len([
                        m for m in self.agent_memory.values() 
                        if (datetime.now() - m.created_at).days < 7
                    ])
                },
                "export_info": {
                    "format": format,
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            if format == "json":
                import json
                return {
                    "success": True,
                    "data": json.loads(json.dumps(export_data, default=datetime_converter)),
                    "format": format
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported export format: {format}",
                    "supported_formats": ["json"]
                }
                
        except Exception as e:
            logger.error(f"Research data export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def import_research_data(self, data: Dict[str, Any]) -> bool:
        """Import research data from external source."""
        try:
            # Validate import data
            if not isinstance(data, dict) or "knowledge_graph" not in data:
                logger.error("Invalid import data format")
                return False
            
            # Import knowledge graph entities
            kg_data = data.get("knowledge_graph", {})
            if "entities" in kg_data:
                for entity_id, entity_data in kg_data["entities"].items():
                    if entity_id not in self.knowledge_graph.entities:
                        # Convert datetime strings back to datetime objects
                        if "created_at" in entity_data and isinstance(entity_data["created_at"], str):
                            entity_data["created_at"] = datetime.fromisoformat(entity_data["created_at"])
                        self.knowledge_graph.entities[entity_id] = entity_data
            
            # Import relationships
            if "relationships" in kg_data:
                for rel_id, rel_data in kg_data["relationships"].items():
                    if rel_id not in self.knowledge_graph.relationships:
                        if "created_at" in rel_data and isinstance(rel_data["created_at"], str):
                            rel_data["created_at"] = datetime.fromisoformat(rel_data["created_at"])
                        self.knowledge_graph.relationships[rel_id] = rel_data
            
            # Import concepts
            if "concepts" in kg_data:
                for concept_id, concept_data in kg_data["concepts"].items():
                    if concept_id not in self.knowledge_graph.concepts:
                        self.knowledge_graph.concepts[concept_id] = concept_data
            
            # Import research sessions (selective)
            sessions_data = data.get("research_sessions", {})
            for session_id, session_data in sessions_data.items():
                if session_id not in self.research_sessions:
                    self.research_sessions[session_id] = session_data
            
            logger.info(f"Successfully imported research data: {len(kg_data.get('entities', {}))} entities, {len(sessions_data)} sessions")
            return True
            
        except Exception as e:
            logger.error(f"Research data import failed: {e}")
            return False
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get comprehensive research statistics."""
        try:
            # Calculate various statistics
            total_memories = len(self.agent_memory)
            memory_by_type = {}
            for memory in self.agent_memory.values():
                memory_type = memory.memory_type
                memory_by_type[memory_type] = memory_by_type.get(memory_type, 0) + 1
            
            # Recent activity (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_memories = len([
                m for m in self.agent_memory.values() 
                if m.created_at > recent_cutoff
            ])
            
            recent_sessions = len([
                s for s in self.research_sessions.values()
                if datetime.fromisoformat(s.get("timestamp", "1970-01-01T00:00:00")) > recent_cutoff
            ])
            
            return {
                "memory_statistics": {
                    "total_memories": total_memories,
                    "memories_by_type": memory_by_type,
                    "recent_memories": recent_memories
                },
                "knowledge_graph_statistics": self.get_knowledge_graph_stats(),
                "research_session_statistics": {
                    "total_sessions": len(self.research_sessions),
                    "recent_sessions": recent_sessions,
                    "active_sessions": len(self.get_active_research_sessions())
                },
                "agent_metrics": self.metrics,
                "capabilities": list(self.capabilities),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
