"""
Ultra AI Project - Creative Agent

Specialized agent for creative writing, storytelling, content generation,
and artistic text creation tasks.
"""

import asyncio
import re
import json
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib

from .base_agent import BaseAgent, AgentConfig, Task, TaskStatus, AgentCapability
from ..utils.logger import get_logger
from ..utils.helpers import sanitize_string, current_timestamp

logger = get_logger(__name__)

class CreativeStyle(Enum):
    """Creative writing styles."""
    NARRATIVE = "narrative"
    DESCRIPTIVE = "descriptive"
    PERSUASIVE = "persuasive"
    EXPOSITORY = "expository"
    POETIC = "poetic"
    DRAMATIC = "dramatic"
    HUMOROUS = "humorous"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    SCIENTIFIC = "scientific"

class ContentType(Enum):
    """Types of creative content."""
    STORY = "story"
    POEM = "poem"
    SCRIPT = "script"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    MARKETING_COPY = "marketing_copy"
    SONG_LYRICS = "song_lyrics"
    SCREENPLAY = "screenplay"
    DIALOGUE = "dialogue"
    CHARACTER_DESCRIPTION = "character_description"
    WORLD_BUILDING = "world_building"

class Genre(Enum):
    """Creative genres."""
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    THRILLER = "thriller"
    HORROR = "horror"
    DRAMA = "drama"
    COMEDY = "comedy"
    ADVENTURE = "adventure"
    HISTORICAL = "historical"
    CONTEMPORARY = "contemporary"
    LITERARY = "literary"

class Character:
    """Character structure for creative writing."""
    
    def __init__(self, name: str, age: Optional[int] = None, occupation: str = "", 
                 personality: List[str] = None, background: str = ""):
        self.name = name
        self.age = age
        self.occupation = occupation
        self.personality = personality or []
        self.background = background
        self.relationships = {}
        self.goals = []
        self.conflicts = []
        self.created_at = datetime.now()

class PlotElement:
    """Plot element structure."""
    
    def __init__(self, element_type: str, description: str, importance: int = 1):
        self.element_type = element_type  # setup, conflict, climax, resolution
        self.description = description
        self.importance = importance
        self.connected_elements = []
        self.characters_involved = []

class CreativeProject:
    """Creative project management."""
    
    def __init__(self, title: str, content_type: ContentType, genre: Genre = None):
        self.project_id = hashlib.md5(f"{title}_{datetime.now()}".encode()).hexdigest()[:12]
        self.title = title
        self.content_type = content_type
        self.genre = genre
        self.characters = {}
        self.plot_elements = []
        self.outline = ""
        self.content = ""
        self.metadata = {}
        self.created_at = datetime.now()
        self.last_modified = datetime.now()
        self.status = "draft"

class CreativeAgent(BaseAgent):
    """Specialized agent for creative writing and content generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Set up creative agent configuration
        if config is None:
            config = {}
        
        agent_config = AgentConfig(
            name=config.get("name", "creative_agent"),
            agent_type="creative",
            max_concurrent_tasks=config.get("max_concurrent_tasks", 3),
            timeout=config.get("timeout", 900.0),  # 15 minutes for creative tasks
            memory_limit=config.get("memory_limit", 1500),
            capabilities=[
                AgentCapability.CREATIVE_WRITING.value,
                AgentCapability.STORYTELLING.value,
                AgentCapability.CONTENT_GENERATION.value,
                AgentCapability.TEXT_GENERATION.value
            ],
            preferred_models=config.get("preferred_models", ["gpt-4", "claude-3-sonnet"]),
            enable_memory=config.get("enable_memory", True),
            custom_settings=config.get("custom_settings", {})
        )
        
        super().__init__(agent_config, **kwargs)
        
        # Creative-specific configuration
        self.creativity_level = config.get("creativity_level", 0.7)  # 0.0 to 1.0
        self.style_consistency = config.get("style_consistency", 0.8)
        self.max_content_length = config.get("max_content_length", 10000)
        self.enable_collaboration = config.get("enable_collaboration", True)
        
        # Creative resources
        self.active_projects = {}
        self.character_library = {}
        self.plot_templates = {}
        self.style_guides = {}
        self.inspiration_sources = []
        
        # Writing statistics
        self.writing_stats = {
            "total_words_generated": 0,
            "projects_completed": 0,
            "favorite_genres": {},
            "style_preferences": {},
            "character_count": 0
        }
        
        logger.info("CreativeAgent initialized")
    
    async def _agent_initialize(self):
        """Creative agent specific initialization."""
        try:
            # Load creative writing templates and resources
            await self._load_creative_templates()
            
            # Initialize style guides
            await self._initialize_style_guides()
            
            # Load character archetypes
            await self._load_character_archetypes()
            
            # Load plot structures
            await self._load_plot_structures()
            
            logger.info("CreativeAgent initialization complete")
            
        except Exception as e:
            logger.error(f"CreativeAgent initialization failed: {e}")
            raise
    
    async def _load_creative_templates(self):
        """Load creative writing templates and frameworks."""
        try:
            templates = {
                "story_structures": {
                    "three_act": {
                        "act1": "Setup - Introduce characters, setting, and initial situation",
                        "act2": "Confrontation - Present obstacles, develop conflict, build tension",
                        "act3": "Resolution - Climax and conclusion, resolve conflicts"
                    },
                    "heros_journey": {
                        "ordinary_world": "Hero's normal life before transformation",
                        "call_to_adventure": "Hero is presented with a challenge",
                        "refusal_of_call": "Hero hesitates or refuses the adventure",
                        "meeting_mentor": "Hero encounters a wise mentor",
                        "crossing_threshold": "Hero commits to the adventure",
                        "tests_allies_enemies": "Hero faces challenges and makes allies",
                        "approach_ordeal": "Hero prepares for major challenge",
                        "ordeal": "Hero confronts their greatest fear",
                        "reward": "Hero survives and gains something",
                        "road_back": "Hero begins journey back to ordinary world",
                        "resurrection": "Hero is transformed by the experience",
                        "return_elixir": "Hero returns with wisdom to help others"
                    },
                    "freytags_pyramid": {
                        "exposition": "Background information and setting",
                        "rising_action": "Series of events that build tension",
                        "climax": "Turning point of the story",
                        "falling_action": "Events after the climax",
                        "resolution": "Conclusion and loose ends tied up"
                    }
                },
                "character_development": {
                    "personality_traits": [
                        "ambitious", "compassionate", "curious", "determined", "empathetic",
                        "honest", "intelligent", "loyal", "optimistic", "resourceful",
                        "stubborn", "impulsive", "cynical", "reckless", "secretive"
                    ],
                    "character_arcs": {
                        "positive_arc": "Character overcomes flaws and grows",
                        "negative_arc": "Character succumbs to flaws and falls",
                        "flat_arc": "Character remains steady and changes the world around them"
                    },
                    "motivations": [
                        "love", "power", "revenge", "survival", "redemption",
                        "discovery", "justice", "freedom", "acceptance", "legacy"
                    ]
                },
                "dialogue_techniques": {
                    "subtext": "What characters mean but don't say directly",
                    "conflict": "Disagreement or tension in conversation",
                    "voice": "Unique speaking patterns for each character",
                    "purpose": "Every dialogue should serve the story",
                    "rhythm": "Varied sentence lengths and pacing"
                }
            }
            
            await self.store_memory(
                content=templates,
                memory_type="creative_templates",
                importance=3.5,
                tags=["templates", "creative_writing", "storytelling"]
            )
            
            self.plot_templates = templates
            
        except Exception as e:
            logger.error(f"Failed to load creative templates: {e}")
    
    async def _initialize_style_guides(self):
        """Initialize writing style guides."""
        try:
            style_guides = {
                "narrative": {
                    "voice": "Third person or first person perspective",
                    "tense": "Past tense typically, present for immediacy",
                    "pacing": "Varied sentence lengths, mix of action and reflection",
                    "description": "Show don't tell, use sensory details"
                },
                "descriptive": {
                    "voice": "Rich, detailed language",
                    "imagery": "Strong sensory descriptions",
                    "metaphors": "Creative comparisons and figurative language",
                    "atmosphere": "Set mood through environmental details"
                },
                "dialogue": {
                    "natural": "Sounds like real speech but more focused",
                    "distinct": "Each character has unique voice",
                    "purposeful": "Advances plot or reveals character",
                    "subtext": "Characters don't always say what they mean"
                },
                "poetic": {
                    "rhythm": "Attention to meter and flow",
                    "imagery": "Vivid and original metaphors",
                    "emotion": "Evokes strong feelings",
                    "compression": "Maximum meaning in minimum words"
                }
            }
            
            await self.store_memory(
                content=style_guides,
                memory_type="style_guides",
                importance=3.0,
                tags=["style", "writing", "guides"]
            )
            
            self.style_guides = style_guides
            
        except Exception as e:
            logger.error(f"Failed to initialize style guides: {e}")
    
    async def _load_character_archetypes(self):
        """Load character archetypes and templates."""
        try:
            archetypes = {
                "hero": {
                    "traits": ["brave", "determined", "moral"],
                    "role": "Protagonist who drives the story forward",
                    "motivations": ["justice", "protection", "duty"],
                    "flaws": ["pride", "recklessness", "naivety"]
                },
                "mentor": {
                    "traits": ["wise", "experienced", "patient"],
                    "role": "Guides and teaches the hero",
                    "motivations": ["legacy", "redemption", "duty"],
                    "flaws": ["secretive", "manipulative", "overprotective"]
                },
                "villain": {
                    "traits": ["cunning", "powerful", "ruthless"],
                    "role": "Primary antagonist opposing the hero",
                    "motivations": ["power", "revenge", "ideology"],
                    "flaws": ["arrogance", "obsession", "cruelty"]
                },
                "ally": {
                    "traits": ["loyal", "supportive", "skilled"],
                    "role": "Supports the hero's journey",
                    "motivations": ["friendship", "common_goal", "debt"],
                    "flaws": ["dependency", "jealousy", "impulsiveness"]
                },
                "trickster": {
                    "traits": ["clever", "unpredictable", "charismatic"],
                    "role": "Brings humor and unpredictability",
                    "motivations": ["fun", "chaos", "personal_gain"],
                    "flaws": ["unreliable", "selfish", "reckless"]
                }
            }

            await self.store_memory(
                content=archetypes,
                memory_type="character_archetypes",
                importance=3.0,
                tags=["characters", "archetypes", "storytelling"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load character archetypes: {e}")
    
    async def _load_plot_structures(self):
        """Load plot structures and narrative frameworks."""
        try:
            plot_structures = {
                "conflict_types": {
                    "person_vs_person": "Character conflicts with another character",
                    "person_vs_self": "Internal struggle within character",
                    "person_vs_society": "Character against social norms or systems",
                    "person_vs_nature": "Character struggles against natural forces",
                    "person_vs_technology": "Character conflicts with technology",
                    "person_vs_supernatural": "Character faces supernatural forces",
                    "person_vs_fate": "Character struggles against destiny"
                },
                "plot_devices": {
                    "foreshadowing": "Hints about future events",
                    "flashback": "Revealing past events",
                    "red_herring": "Misleading clue or information",
                    "deus_ex_machina": "Unexpected resolution to conflict",
                    "chekovs_gun": "Element introduced early must be used later",
                    "cliffhanger": "Suspenseful ending that demands continuation"
                },
                "pacing_techniques": {
                    "action_scenes": "Fast-paced, short sentences, immediate danger",
                    "emotional_scenes": "Slower pace, internal reflection, deeper character development",
                    "exposition": "Moderate pace, world-building and information delivery",
                    "dialogue_scenes": "Character-driven pace, relationship development"
                }
            }
            
            await self.store_memory(
                content=plot_structures,
                memory_type="plot_structures",
                importance=3.0,
                tags=["plot", "structure", "narrative"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load plot structures: {e}")
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a creative writing task."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "create_story":
                return await self._create_story(data)
            elif task_type == "write_poem":
                return await self._write_poem(data)
            elif task_type == "generate_content":
                return await self._generate_content(data)
            elif task_type == "develop_character":
                return await self._develop_character(data)
            elif task_type == "create_dialogue":
                return await self._create_dialogue(data)
            elif task_type == "write_script":
                return await self._write_script(data)
            elif task_type == "create_outline":
                return await self._create_outline(data)
            elif task_type == "rewrite_content":
                return await self._rewrite_content(data)
            elif task_type == "continue_story":
                return await self._continue_story(data)
            elif task_type == "brainstorm_ideas":
                return await self._brainstorm_ideas(data)
            elif task_type == "create_world":
                return await self._create_world(data)
            elif task_type == "write_marketing_copy":
                return await self._write_marketing_copy(data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _create_story(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete story."""
        try:
            prompt = data.get("prompt", "")
            genre = data.get("genre", Genre.CONTEMPORARY.value)
            style = data.get("style", CreativeStyle.NARRATIVE.value)
            length = data.get("length", "medium")  # short, medium, long
            characters = data.get("characters", [])
            setting = data.get("setting", "")
            theme = data.get("theme", "")
            
            if not prompt:
                raise ValueError("Story prompt is required")
            
            # Create a new project
            project = CreativeProject(
                title=data.get("title", f"Story: {prompt[:50]}..."),
                content_type=ContentType.STORY,
                genre=Genre(genre) if genre in [g.value for g in Genre] else Genre.CONTEMPORARY
            )
            
            # Determine story length parameters
            length_params = {
                "short": {"target_words": 500, "max_tokens": 800},
                "medium": {"target_words": 1500, "max_tokens": 2000},
                "long": {"target_words": 3000, "max_tokens": 4000}
            }
            
            params = length_params.get(length, length_params["medium"])
            
            # Get creative templates for inspiration
            templates = await self.retrieve_memory(memory_type="creative_templates", limit=1)
            
            # Build story creation prompt
            story_prompt = await self._build_story_prompt(
                prompt, genre, style, length, characters, setting, theme, templates
            )
            
            # Generate the story
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=story_prompt,
                temperature=self.creativity_level,
                max_tokens=params["max_tokens"]
            )
            
            if not response.success:
                raise ValueError(f"Story generation failed: {response.error}")
            
            story_content = response.content
            
            # Update project
            project.content = story_content
            project.metadata = {
                "prompt": prompt,
                "genre": genre,
                "style": style,
                "length": length,
                "word_count": len(story_content.split()),
                "characters": characters,
                "setting": setting,
                "theme": theme
            }
            project.status = "completed"
            project.last_modified = datetime.now()
            
            # Store project
            self.active_projects[project.project_id] = project
            
            # Update statistics
            self.writing_stats["total_words_generated"] += len(story_content.split())
            self.writing_stats["projects_completed"] += 1
            if genre in self.writing_stats["favorite_genres"]:
                self.writing_stats["favorite_genres"][genre] += 1
            else:
                self.writing_stats["favorite_genres"][genre] = 1
            
            result = {
                "project_id": project.project_id,
                "title": project.title,
                "story": story_content,
                "metadata": project.metadata,
                "word_count": len(story_content.split()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="creative_work",
                importance=3.0,
                tags=["story", genre, style, length]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Story creation failed: {e}")
            raise
    
    async def _build_story_prompt(self, prompt: str, genre: str, style: str, length: str,
                                characters: List[str], setting: str, theme: str, templates: List) -> str:
        """Build a comprehensive prompt for story generation."""
        
        story_prompt = f"""Write a {length} {genre} story in {style} style based on this prompt:

"{prompt}"
"""
        
        if setting:
            story_prompt += f"\nSetting: {setting}"
        
        if characters:
            story_prompt += f"\nCharacters to include: {', '.join(characters)}"
        
        if theme:
            story_prompt += f"\nTheme: {theme}"
        
        # Add creative guidelines
        story_prompt += f"""

Creative Guidelines:
- Write in {style} style with appropriate tone and voice
- Include rich, sensory descriptions
- Develop characters with distinct personalities and motivations
- Create engaging dialogue that reveals character
- Build tension and conflict appropriate to the {genre} genre
- Use show-don't-tell techniques
- Create a satisfying narrative arc"""
        
        if length == "short":
            story_prompt += "\n- Focus on a single scene or moment with high impact"
        elif length == "medium":
            story_prompt += "\n- Develop a complete story arc with beginning, middle, and end"
        elif length == "long":
            story_prompt += "\n- Create multiple scenes with character development and subplot elements"
        
        # Add genre-specific elements
        genre_elements = {
            "fantasy": "Include magical elements, mythical creatures, or supernatural powers",
            "science_fiction": "Incorporate futuristic technology, space travel, or scientific concepts",
            "mystery": "Create suspense, clues, and a puzzle to be solved",
            "romance": "Focus on emotional connections and relationship development",
            "horror": "Build atmosphere of fear, suspense, and dread",
            "thriller": "Maintain high tension and fast-paced action"
        }
        
        if genre in genre_elements:
            story_prompt += f"\n- {genre_elements[genre]}"
        
        story_prompt += "\n\nBegin the story:"
        
        return story_prompt
    
    async def _write_poem(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write a poem based on specifications."""
        try:
            theme = data.get("theme", "")
            style = data.get("style", "free_verse")  # free_verse, sonnet, haiku, ballad
            tone = data.get("tone", "reflective")  # reflective, joyful, melancholic, passionate
            length = data.get("length", "medium")  # short, medium, long
            rhyme_scheme = data.get("rhyme_scheme", "none")
            
            if not theme:
                raise ValueError("Poem theme is required")
            
            # Build poem prompt
            poem_prompt = f"""Write a {style} poem about {theme}.

Style: {style}
Tone: {tone}
Length: {length}
"""
            
            if rhyme_scheme != "none":
                poem_prompt += f"Rhyme scheme: {rhyme_scheme}\n"
            
            poem_prompt += f"""
Poetry Guidelines:
- Create vivid, original imagery
- Use metaphors and similes effectively
- Pay attention to rhythm and flow
- Evoke strong emotions related to the {tone} tone
- Use sensory language and specific details
- Make every word count - be precise and impactful
"""
            
            # Style-specific instructions
            style_instructions = {
                "sonnet": "Follow traditional sonnet structure with 14 lines and iambic pentameter",
                "haiku": "Write three lines with 5-7-5 syllable pattern, focus on nature or moment",
                "ballad": "Tell a story through verse, use narrative elements",
                "free_verse": "Focus on natural speech rhythms and imagery without formal constraints"
            }
            
            if style in style_instructions:
                poem_prompt += f"\n- {style_instructions[style]}"
            
            poem_prompt += "\n\nWrite the poem:"
            
            # Generate poem
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=poem_prompt,
                temperature=self.creativity_level + 0.1,  # Slightly higher creativity for poetry
                max_tokens=500
            )
            
            if not response.success:
                raise ValueError(f"Poem generation failed: {response.error}")
            
            poem_content = response.content
            
            # Analyze poem structure
            lines = poem_content.strip().split('\n')
            word_count = len(poem_content.split())
            
            result = {
                "poem": poem_content,
                "theme": theme,
                "style": style,
                "tone": tone,
                "length": length,
                "rhyme_scheme": rhyme_scheme,
                "analysis": {
                    "line_count": len([line for line in lines if line.strip()]),
                    "word_count": word_count,
                    "estimated_reading_time": f"{max(1, word_count // 200)} minute(s)"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.writing_stats["total_words_generated"] += word_count
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="creative_work",
                importance=2.5,
                tags=["poem", style, tone, theme.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Poem writing failed: {e}")
            raise
    
    async def _generate_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on type and specifications."""
        try:
            content_type = data.get("content_type", ContentType.ARTICLE.value)
            topic = data.get("topic", "")
            audience = data.get("audience", "general")
            tone = data.get("tone", "professional")
            length = data.get("length", "medium")
            purpose = data.get("purpose", "inform")  # inform, persuade, entertain, educate
            
            if not topic:
                raise ValueError("Content topic is required")
            
            # Length parameters for different content types
            length_params = {
                "short": {"words": 300, "tokens": 500},
                "medium": {"words": 800, "tokens": 1200},
                "long": {"words": 1500, "tokens": 2000}
            }
            
            params = length_params.get(length, length_params["medium"])
            
            # Build content generation prompt
            content_prompt = await self._build_content_prompt(
                content_type, topic, audience, tone, length, purpose
            )
            
            # Generate content
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=content_prompt,
                temperature=self.creativity_level,
                max_tokens=params["tokens"]
            )
            
            if not response.success:
                raise ValueError(f"Content generation failed: {response.error}")
            
            generated_content = response.content
            word_count = len(generated_content.split())
            
            result = {
                "content": generated_content,
                "content_type": content_type,
                "topic": topic,
                "audience": audience,
                "tone": tone,
                "length": length,
                "purpose": purpose,
                "word_count": word_count,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.writing_stats["total_words_generated"] += word_count
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="creative_work",
                importance=2.0,
                tags=["content", content_type, purpose, tone]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise
    
    async def _build_content_prompt(self, content_type: str, topic: str, audience: str,
                                  tone: str, length: str, purpose: str) -> str:
        """Build prompt for content generation."""
        
        content_prompt = f"""Create {length} {content_type} content about "{topic}".

Target Audience: {audience}
Tone: {tone}
Purpose: {purpose}
"""
        
        # Content type specific guidelines
        type_guidelines = {
            "article": "Structure with clear introduction, body paragraphs, and conclusion. Use headings and subheadings.",
            "blog_post": "Write in conversational tone, include engaging hook, use personal examples where appropriate.",
            "social_media": "Create engaging, shareable content with strong hook. Be concise and impactful.",
            "marketing_copy": "Focus on benefits, include call-to-action, address customer pain points.",
            "script": "Write in proper script format with character names, dialogue, and stage directions.",
            "newsletter": "Use friendly tone, include multiple sections, make it scannable with bullet points."
        }
        
        if content_type in type_guidelines:
            content_prompt += f"\nContent Guidelines:\n- {type_guidelines[content_type]}"
        
        # Audience-specific adjustments
        audience_guidelines = {
            "children": "Use simple language, engaging examples, avoid complex concepts",
            "teenagers": "Use contemporary language, relevant examples, energetic tone",
            "professionals": "Use industry terminology, focus on practical applications, be concise",
            "academics": "Use formal language, include research references, detailed analysis",
            "general": "Use clear, accessible language, universal examples, balanced approach"
        }
        
        if audience in audience_guidelines:
            content_prompt += f"\n- {audience_guidelines[audience]}"
        
        # Purpose-specific elements
        purpose_elements = {
            "inform": "Present facts clearly, use credible sources, maintain objectivity",
            "persuade": "Use compelling arguments, include evidence, address counterarguments",
            "entertain": "Use humor, storytelling, engaging anecdotes, keep readers interested",
            "educate": "Break down complex concepts, use examples, include practical applications"
        }
        
        if purpose in purpose_elements:
            content_prompt += f"\n- {purpose_elements[purpose]}"
        
        content_prompt += f"\n\nWrite the {content_type}:"
        
        return content_prompt
    
    async def _develop_character(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop a detailed character."""
        try:
            name = data.get("name", "")
            role = data.get("role", "protagonist")  # protagonist, antagonist, supporting
            archetype = data.get("archetype", "")
            age_range = data.get("age_range", "adult")
            background = data.get("background", "")
            personality_traits = data.get("personality_traits", [])
            goals = data.get("goals", [])
            
            if not name:
                # Generate a name if not provided
                name = await self._generate_character_name()
            
            # Get character archetypes for reference
            archetypes = await self.retrieve_memory(memory_type="character_archetypes", limit=1)
            
            # Build character development prompt
            character_prompt = f"""Create a detailed character profile for {name}, a {role} character.

Character Details:
- Name: {name}
- Role: {role}
- Age Range: {age_range}
"""
            
            if archetype:
                character_prompt += f"- Archetype: {archetype}\n"
            
            if background:
                character_prompt += f"- Background: {background}\n"
            
            if personality_traits:
                character_prompt += f"- Personality Traits: {', '.join(personality_traits)}\n"
            
            if goals:
                character_prompt += f"- Goals: {', '.join(goals)}\n"
            
            character_prompt += """
Please provide:
1. Physical description
2. Detailed personality traits and quirks
3. Background and history
4. Motivations and goals
5. Fears and weaknesses
6. Relationships with others
7. Character arc potential
8. Unique voice and speech patterns
9. Internal conflicts
10. Role in the story

Make the character feel real, complex, and three-dimensional."""
            
            # Generate character
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=character_prompt,
                temperature=self.creativity_level,
                max_tokens=1000
            )
            
            if not response.success:
                raise ValueError(f"Character development failed: {response.error}")
            
            character_description = response.content
            
            # Create character object
            character = Character(
                name=name,
                age=None,  # Would extract from description if needed
                occupation="",  # Would extract from description
                personality=personality_traits,
                background=background
            )
            
            if goals:
                character.goals = goals
            
            # Store character in library
            character_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()[:12]
            self.character_library[character_id] = character
            
            result = {
                "character_id": character_id,
                "name": name,
                "role": role,
                "archetype": archetype,
                "description": character_description,
                "metadata": {
                    "age_range": age_range,
                    "background": background,
                    "personality_traits": personality_traits,
                    "goals": goals
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Update statistics
            self.writing_stats["character_count"] += 1
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="character_development",
                importance=2.5,
                tags=["character", role, archetype, name.lower().replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Character development failed: {e}")
            raise
    
    async def _generate_character_name(self) -> str:
        """Generate a character name."""
        first_names = [
            "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
            "Elena", "Marcus", "Sophia", "Ethan", "Isabella", "Noah", "Emma", "Liam",
            "Aria", "Kai", "Luna", "Felix", "Nova", "Sage", "River", "Phoenix"
        ]
        
        last_names = [
            "Chen", "Rodriguez", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Martinez", "Anderson", "Taylor", "Thomas", "Moore",
            "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Clark"
        ]
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    async def _create_dialogue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create dialogue between characters."""
        try:
            characters = data.get("characters", [])
            scenario = data.get("scenario", "")
            tone = data.get("tone", "natural")
            purpose = data.get("purpose", "character_development")  # plot_advancement, character_development, conflict
            length = data.get("length", "medium")
            
            if len(characters) < 2:
                raise ValueError("At least two characters required for dialogue")
            
            if not scenario:
                raise ValueError("Scenario or context is required for dialogue")
            
            # Build dialogue prompt
            dialogue_prompt = f"""Create a dialogue between {', '.join(characters)} in the following scenario:

Scenario: {scenario}

Dialogue Parameters:
- Tone: {tone}
- Purpose: {purpose}
- Length: {length}

Guidelines:
- Give each character a distinct voice and speaking style
- Make the dialogue natural but purposeful
- Include subtext and character motivations
- Use appropriate tags and actions sparingly
- Advance the scene through conversation
- Show character relationships through dialogue
"""
            
            if purpose == "conflict":
                dialogue_prompt += "\n- Create tension and disagreement between characters"
            elif purpose == "character_development":
                dialogue_prompt += "\n- Reveal character traits, backgrounds, and motivations"
            elif purpose == "plot_advancement":
                dialogue_prompt += "\n- Move the story forward through information or decisions"
            
            dialogue_prompt += "\n\nWrite the dialogue:"
            
            # Generate dialogue
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=dialogue_prompt,
                temperature=self.creativity_level,
                max_tokens=800
            )
            
            if not response.success:
                raise ValueError(f"Dialogue creation failed: {response.error}")
            
            dialogue_content = response.content
            
            result = {
                "dialogue": dialogue_content,
                "characters": characters,
                "scenario": scenario,
                "tone": tone,
                "purpose": purpose,
                "length": length,
                "word_count": len(dialogue_content.split()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="dialogue",
                importance=2.0,
                tags=["dialogue", tone, purpose] + [char.lower().replace(" ", "_") for char in characters]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Dialogue creation failed: {e}")
            raise
    
    async def _write_script(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write a script for theater, film, or other media."""
        try:
            script_type = data.get("script_type", "short_film")  # short_film, play, monologue, commercial
            premise = data.get("premise", "")
            characters = data.get("characters", [])
            setting = data.get("setting", "")
            length = data.get("length", "short")  # short, medium, full
            genre = data.get("genre", "drama")
            
            if not premise:
                raise ValueError("Script premise is required")
            
            # Build script prompt
            script_prompt = f"""Write a {script_type} script based on this premise:

{premise}

Script Details:
- Type: {script_type}
- Genre: {genre}
- Length: {length}
- Setting: {setting}
"""
            
            if characters:
                script_prompt += f"- Characters: {', '.join(characters)}\n"
            
            script_prompt += f"""
Script Format Guidelines:
- Use proper script formatting with character names in ALL CAPS
- Include stage directions in parentheses
- Write natural, speakable dialogue
- Include scene descriptions and settings
- Use appropriate script conventions for {script_type}
"""
            
            # Script type specific guidelines
            type_guidelines = {
                "short_film": "Focus on visual storytelling, minimal dialogue, strong opening and ending",
                "play": "Emphasize dialogue and character interaction, consider stage limitations",
                "monologue": "Single character piece, internal conflict, personal revelation",
                "commercial": "Brief, persuasive, clear call-to-action, memorable hook"
            }
            
            if script_type in type_guidelines:
                script_prompt += f"\n- {type_guidelines[script_type]}"
            
            script_prompt += "\n\nWrite the script:"
            
            # Generate script
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            length_tokens = {
                "short": 600,
                "medium": 1200,
                "full": 2000
            }
            
            response = await self.model_manager.generate_completion(
                prompt=script_prompt,
                temperature=self.creativity_level,
                max_tokens=length_tokens.get(length, 600)
            )
            
            if not response.success:
                raise ValueError(f"Script writing failed: {response.error}")
            
            script_content = response.content
            
            # Analyze script structure
            lines = script_content.split('\n')
            dialogue_lines = len([line for line in lines if ':' in line])
            stage_directions = len([line for line in lines if '(' in line and ')' in line])
            
            result = {
                "script": script_content,
                "script_type": script_type,
                "premise": premise,
                "genre": genre,
                "length": length,
                "characters": characters,
                "setting": setting,
                "analysis": {
                    "total_lines": len(lines),
                    "dialogue_lines": dialogue_lines,
                    "stage_directions": stage_directions,
                    "word_count": len(script_content.split())
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="script",
                importance=2.5,
                tags=["script", script_type, genre, length]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Script writing failed: {e}")
            raise
    
    async def _create_outline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an outline for a creative work."""
        try:
            content_type = data.get("content_type", "story")
            theme = data.get("theme", "")
            genre = data.get("genre", "")
            length = data.get("length", "medium")
            structure = data.get("structure", "three_act")  # three_act, heros_journey, freytags_pyramid
            
            if not theme:
                raise ValueError("Theme or main idea is required for outline")
            
            # Get plot structures for reference
            structures = await self.retrieve_memory(memory_type="creative_templates", limit=1)
            
            # Build outline prompt
            outline_prompt = f"""Create a detailed outline for a {content_type} with the following specifications:

Theme: {theme}
Genre: {genre}
Length: {length}
Structure: {structure}

Outline Requirements:
- Break down into clear sections or acts
- Include character development arcs
- Identify major plot points and conflicts
- Note pacing and tension building
- Include beginning, middle, and end structure
- Specify key scenes or chapters
"""
            
            if structure == "three_act":
                outline_prompt += """
- Act I: Setup (25%) - Introduce characters, world, inciting incident
- Act II: Confrontation (50%) - Rising action, obstacles, midpoint
- Act III: Resolution (25%) - Climax, falling action, conclusion"""
            elif structure == "heros_journey":
                outline_prompt += """
- Follow the hero's journey structure with departure, initiation, and return
- Include call to adventure, mentor, tests, ordeal, and transformation"""
            
            outline_prompt += "\n\nCreate the outline:"
            
            # Generate outline
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=outline_prompt,
                temperature=self.creativity_level - 0.1,  # Slightly lower for structure
                max_tokens=1000
            )
            
            if not response.success:
                raise ValueError(f"Outline creation failed: {response.error}")
            
            outline_content = response.content
            
            result = {
                "outline": outline_content,
                "content_type": content_type,
                "theme": theme,
                "genre": genre,
                "length": length,
                "structure": structure,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="outline",
                importance=2.0,
                tags=["outline", content_type, structure, genre]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Outline creation failed: {e}")
            raise
    
    async def _rewrite_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite existing content with new specifications."""
        try:
            original_content = data.get("original_content", "")
            rewrite_type = data.get("rewrite_type", "improve")  # improve, change_style, change_tone, expand, condense
            target_style = data.get("target_style", "")
            target_tone = data.get("target_tone", "")
            target_length = data.get("target_length", "same")
            specific_instructions = data.get("instructions", "")
            
            if not original_content:
                raise ValueError("Original content is required for rewriting")
            
            # Build rewrite prompt
            rewrite_prompt = f"""Rewrite the following content according to the specifications:

Original Content:
{original_content}

Rewrite Specifications:
- Type: {rewrite_type}
"""
            
            if target_style:
                rewrite_prompt += f"- Target Style: {target_style}\n"
            
            if target_tone:
                rewrite_prompt += f"- Target Tone: {target_tone}\n"
            
            if target_length != "same":
                rewrite_prompt += f"- Target Length: {target_length}\n"
            
            if specific_instructions:
                rewrite_prompt += f"- Additional Instructions: {specific_instructions}\n"
            
            # Add rewrite type specific instructions
            rewrite_instructions = {
                "improve": "Enhance clarity, flow, and impact while maintaining the original meaning",
                "change_style": "Adapt the writing style while preserving the content and message",
                "change_tone": "Adjust the emotional tone and voice while keeping the same information",
                "expand": "Add more detail, examples, and depth to the existing content",
                "condense": "Shorten while retaining all essential information and impact"
            }
            
            if rewrite_type in rewrite_instructions:
                rewrite_prompt += f"\nRewrite Guidelines:\n- {rewrite_instructions[rewrite_type]}"
            
            rewrite_prompt += "\n\nProvide the rewritten content:"
            
            # Generate rewrite
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=rewrite_prompt,
                temperature=self.creativity_level,
                max_tokens=1500
            )
            
            if not response.success:
                raise ValueError(f"Content rewriting failed: {response.error}")
            
            rewritten_content = response.content
            
            # Compare original and rewritten content
            original_word_count = len(original_content.split())
            rewritten_word_count = len(rewritten_content.split())
            
            result = {
                "original_content": original_content,
                "rewritten_content": rewritten_content,
                "rewrite_type": rewrite_type,
                "target_style": target_style,
                "target_tone": target_tone,
                "target_length": target_length,
                "instructions": specific_instructions,
                "comparison": {
                    "original_word_count": original_word_count,
                    "rewritten_word_count": rewritten_word_count,
                    "length_change": f"{((rewritten_word_count / original_word_count - 1) * 100):.1f}%"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="rewrite",
                importance=2.0,
                tags=["rewrite", rewrite_type, target_style, target_tone]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Content rewriting failed: {e}")
            raise
    
    async def _continue_story(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Continue an existing story."""
        try:
            existing_story = data.get("existing_story", "")
            direction = data.get("direction", "")
            style_consistency = data.get("maintain_style", True)
            length = data.get("length", "medium")
            
            if not existing_story:
                raise ValueError("Existing story content is required")
            
            # Analyze existing story for style and context
            story_analysis = await self._analyze_story_style(existing_story)
            
            # Build continuation prompt
            continue_prompt = f"""Continue the following story:

{existing_story}

Continuation Guidelines:
- Maintain the same narrative voice and style
- Keep character personalities consistent
- Continue the established tone and mood
- Build naturally from where the story left off
"""
            
            if style_consistency:
                continue_prompt += f"""
- Match the writing style: {story_analysis.get('style', 'narrative')}
- Maintain the same tense and perspective
- Use similar sentence structure and vocabulary"""
            
            if direction:
                continue_prompt += f"\n- Story direction: {direction}"
            
            length_guidelines = {
                "short": "Add 1-2 paragraphs to advance the story",
                "medium": "Continue with 3-5 paragraphs developing the plot",
                "long": "Write an extended continuation with multiple scenes"
            }
            
            continue_prompt += f"\n- Length: {length_guidelines.get(length, length_guidelines['medium'])}"
            continue_prompt += "\n\nContinue the story:"
            
            # Generate continuation
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            length_tokens = {
                "short": 300,
                "medium": 600,
                "long": 1000
            }
            
            response = await self.model_manager.generate_completion(
                prompt=continue_prompt,
                temperature=self.style_consistency,  # Use style consistency as temperature modifier
                max_tokens=length_tokens.get(length, 600)
            )
            
            if not response.success:
                raise ValueError(f"Story continuation failed: {response.error}")
            
            continuation = response.content
            full_story = existing_story + "\n\n" + continuation
            
            result = {
                "original_story": existing_story,
                "continuation": continuation,
                "full_story": full_story,
                "direction": direction,
                "style_analysis": story_analysis,
                "length": length,
                "word_count_added": len(continuation.split()),
                "total_word_count": len(full_story.split()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="story_continuation",
                importance=2.5,
                tags=["continuation", "story", length]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Story continuation failed: {e}")
            raise
    
    async def _analyze_story_style(self, story: str) -> Dict[str, Any]:
        """Analyze the style of an existing story."""
        try:
            # Simple style analysis - could be enhanced with NLP
            sentences = story.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Determine narrative perspective
            perspective = "third_person"
            if " I " in story or story.startswith("I "):
                perspective = "first_person"
            elif " you " in story or story.startswith("You "):
                perspective = "second_person"
            
            # Determine tense
            tense = "past"
            if " is " in story or " are " in story:
                tense = "present"
            
            # Estimate style complexity
            complexity = "simple"
            if avg_sentence_length > 15:
                complexity = "complex"
            elif avg_sentence_length > 10:
                complexity = "moderate"
            
            return {
                "perspective": perspective,
                "tense": tense,
                "complexity": complexity,
                "avg_sentence_length": avg_sentence_length,
                "style": "narrative"  # Default to narrative
            }
            
        except Exception as e:
            logger.error(f"Story style analysis failed: {e}")
            return {"style": "narrative", "perspective": "third_person", "tense": "past"}
    
    async def _brainstorm_ideas(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Brainstorm creative ideas based on parameters."""
        try:
            topic = data.get("topic", "")
            idea_type = data.get("idea_type", "story")  # story, character, plot, setting, theme
            genre = data.get("genre", "")
            quantity = data.get("quantity", 10)
            creativity_level = data.get("creativity_level", "medium")  # low, medium, high
            
            if not topic:
                raise ValueError("Topic or theme is required for brainstorming")
            
            # Build brainstorming prompt
            brainstorm_prompt = f"""Brainstorm {quantity} creative {idea_type} ideas related to "{topic}".

Parameters:
- Type: {idea_type}
- Genre: {genre}
- Creativity Level: {creativity_level}

Guidelines:
- Provide diverse and original ideas
- Include brief descriptions for each idea
- Make ideas specific and actionable
- Consider different angles and perspectives
"""
            
            if creativity_level == "high":
                brainstorm_prompt += "\n- Push boundaries, think outside the box, combine unexpected elements"
            elif creativity_level == "medium":
                brainstorm_prompt += "\n- Balance familiar concepts with fresh twists"
            else:
                brainstorm_prompt += "\n- Focus on solid, proven concepts with minor variations"
            
            # Add type-specific instructions
            type_instructions = {
                "story": "Include premise, main conflict, and potential resolution",
                "character": "Include role, personality traits, and backstory elements",
                "plot": "Include setup, conflict, and resolution structure",
                "setting": "Include time period, location, and unique features",
                "theme": "Include central message and how it could be explored"
            }
            
            if idea_type in type_instructions:
                brainstorm_prompt += f"\n- {type_instructions[idea_type]}"
            
            brainstorm_prompt += f"\n\nGenerate {quantity} ideas:"
            
            # Generate ideas
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            creativity_temperature = {
                "low": 0.4,
                "medium": 0.7,
                "high": 0.9
            }
            
            response = await self.model_manager.generate_completion(
                prompt=brainstorm_prompt,
                temperature=creativity_temperature.get(creativity_level, 0.7),
                max_tokens=1200
            )
            
            if not response.success:
                raise ValueError(f"Idea brainstorming failed: {response.error}")
            
            ideas_content = response.content
            
            # Parse ideas from response
            ideas = self._parse_brainstormed_ideas(ideas_content)
            
            result = {
                "topic": topic,
                "idea_type": idea_type,
                "genre": genre,
                "creativity_level": creativity_level,
                "requested_quantity": quantity,
                "generated_ideas": ideas,
                "actual_quantity": len(ideas),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="brainstorming",
                importance=2.0,
                tags=["brainstorming", idea_type, creativity_level, topic.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Idea brainstorming failed: {e}")
            raise
    
    def _parse_brainstormed_ideas(self, content: str) -> List[Dict[str, str]]:
        """Parse brainstormed ideas from response content."""
        ideas = []
        lines = content.split('\n')
        
        current_idea = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number or bullet point
            if re.match(r'^[\d]+\.?\s+', line) or line.startswith(('- ', '* ')):
                # New idea
                if current_idea:
                    ideas.append(current_idea)
                
                # Extract title
                title = re.sub(r'^[\d\-\*\.]+\s*', '', line).strip()
                current_idea = {"title": title, "description": ""}
            elif current_idea and line:
                # Continuation of description
                current_idea["description"] += " " + line if current_idea["description"] else line
        
        # Add last idea
        if current_idea:
            ideas.append(current_idea)
        
        return ideas
    
    async def _create_world(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed fictional world or setting."""
        try:
            world_type = data.get("world_type", "fantasy")  # fantasy, sci-fi, modern, historical
            scale = data.get("scale", "city")  # room, building, city, region, continent, planet
            key_features = data.get("key_features", [])
            time_period = data.get("time_period", "")
            culture_type = data.get("culture_type", "")
            
            # Build world creation prompt
            world_prompt = f"""Create a detailed {world_type} world/setting at the {scale} scale.

World Parameters:
- Type: {world_type}
- Scale: {scale}
- Time Period: {time_period}
"""
            
            if key_features:
                world_prompt += f"- Key Features: {', '.join(key_features)}\n"
            
            if culture_type:
                world_prompt += f"- Culture Type: {culture_type}\n"
            
            world_prompt += f"""
World Building Elements to Include:
- Physical geography and environment
- Climate and weather patterns
- Architecture and infrastructure
- Social structure and governance
- Economy and trade
- Culture and customs
- History and background
- Notable locations or landmarks
- Potential conflicts or tensions
- Unique features that make it memorable

Make the world feel lived-in, consistent, and rich with detail."""
            
            # Generate world description
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=world_prompt,
                temperature=self.creativity_level,
                max_tokens=1500
            )
            
            if not response.success:
                raise ValueError(f"World creation failed: {response.error}")
            
            world_description = response.content
            
            result = {
                "world_description": world_description,
                "world_type": world_type,
                "scale": scale,
                "key_features": key_features,
                "time_period": time_period,
                "culture_type": culture_type,
                "word_count": len(world_description.split()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="world_building",
                importance=3.0,
                tags=["world_building", world_type, scale, culture_type]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"World creation failed: {e}")
            raise
    
    async def _write_marketing_copy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write marketing copy for products or services."""
        try:
            product_service = data.get("product_service", "")
            target_audience = data.get("target_audience", "")
            copy_type = data.get("copy_type", "advertisement")  # advertisement, email, landing_page, social_media
            tone = data.get("tone", "persuasive")
            key_benefits = data.get("key_benefits", [])
            call_to_action = data.get("call_to_action", "")
            length = data.get("length", "medium")
            
            if not product_service:
                raise ValueError("Product or service description is required")
            
Marketing Parameters:
- Target Audience: {target_audience}
- Copy Type: {copy_type}
- Tone: {tone}
- Length: {length}
"""
            
            if key_benefits:
                copy_prompt += f"- Key Benefits: {', '.join(key_benefits)}\n"
            
            if call_to_action:
                copy_prompt += f"- Call to Action: {call_to_action}\n"
            
            copy_prompt += f"""
Marketing Copy Guidelines:
- Hook readers with compelling opening
- Focus on benefits over features
- Address customer pain points
- Use persuasive language and social proof
- Include clear call-to-action
- Match the {tone} tone throughout
- Optimize for {copy_type} format
"""
            
            # Add copy type specific guidelines
            type_guidelines = {
                "advertisement": "Create attention-grabbing headline, concise body, strong CTA",
                "email": "Compelling subject line, personal tone, clear value proposition",
                "landing_page": "Hero section, benefits list, testimonials, CTA",
                "social_media": "Engaging hook, hashtags, shareable content"
            }
            
            if copy_type in type_guidelines:
                copy_prompt += f"\n- {type_guidelines[copy_type]}"
            
            copy_prompt += "\n\nWrite the marketing copy:"
            
            # Generate marketing copy
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            length_tokens = {
                "short": 200,
                "medium": 400,
                "long": 800
            }
            
            response = await self.model_manager.generate_completion(
                prompt=copy_prompt,
                temperature=self.creativity_level - 0.1,  # Slightly lower for marketing
                max_tokens=length_tokens.get(length, 400)
            )
            
            if not response.success:
                raise ValueError(f"Marketing copy writing failed: {response.error}")
            
            marketing_copy = response.content
            
            result = {
                "marketing_copy": marketing_copy,
                "product_service": product_service,
                "target_audience": target_audience,
                "copy_type": copy_type,
                "tone": tone,
                "key_benefits": key_benefits,
                "call_to_action": call_to_action,
                "length": length,
                "word_count": len(marketing_copy.split()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory
            await self.store_memory(
                content=result,
                memory_type="marketing_copy",
                importance=2.0,
                tags=["marketing", copy_type, tone, target_audience.replace(" ", "_")]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Marketing copy writing failed: {e}")
            raise
    
    async def _agent_maintenance(self):
        """Creative agent specific maintenance tasks."""
        try:
            # Update writing statistics
            await self._update_writing_statistics()
            
            # Clean up old projects
            await self._cleanup_old_projects()
            
            # Analyze creative trends
            await self._analyze_creative_trends()
            
            # Optimize character library
            await self._optimize_character_library()
            
except Exception as e:
            logger.error(f"Creative agent maintenance failed: {e}")
    
    async def _update_writing_statistics(self):
        """Update writing statistics and performance metrics."""
        try:
            # Calculate recent productivity
            recent_memories = await self.retrieve_memory(
                memory_type="creative_work",
                limit=50
            )
            
            recent_word_count = sum(
                memory.content.get("word_count", 0) 
                for memory in recent_memories 
                if (datetime.now() - memory.created_at).days < 7
            )
            
            # Update statistics
            self.writing_stats["recent_productivity"] = recent_word_count
            self.writing_stats["avg_words_per_day"] = recent_word_count / 7
            
            # Analyze genre preferences
            genre_count = {}
            for memory in recent_memories:
                content = memory.content
                genre = content.get("genre") or content.get("content_type", "unknown")
                genre_count[genre] = genre_count.get(genre, 0) + 1
            
            self.writing_stats["recent_genres"] = genre_count
            
            logger.debug(f"Updated writing statistics: {recent_word_count} words in last 7 days")
            
        except Exception as e:
            logger.error(f"Writing statistics update failed: {e}")
    
    async def _cleanup_old_projects(self):
        """Clean up old creative projects."""
        try:
            # Remove projects older than 90 days that are marked as draft
            cutoff_date = datetime.now() - timedelta(days=90)
            
            projects_to_remove = []
            for project_id, project in self.active_projects.items():
                if (project.created_at < cutoff_date and 
                    project.status == "draft" and 
                    not project.content):  # Empty drafts
                    projects_to_remove.append(project_id)
            
            # Remove old draft projects
            for project_id in projects_to_remove[:20]:  # Limit removal
                if project_id in self.active_projects:
                    del self.active_projects[project_id]
            
            if projects_to_remove:
                logger.debug(f"Cleaned up {len(projects_to_remove)} old draft projects")
                
        except Exception as e:
            logger.error(f"Project cleanup failed: {e}")
    
    async def _analyze_creative_trends(self):
        """Analyze creative writing trends and patterns."""
        try:
            # Analyze recent creative works for trends
            recent_works = await self.retrieve_memory(
                memory_type="creative_work",
                limit=30
            )
            
            if not recent_works:
                return
            
            # Identify trending themes
            theme_count = {}
            style_count = {}
            
            for work in recent_works:
                content = work.content
                
                # Count themes (simplified)
                theme = content.get("theme") or content.get("topic", "")
                if theme:
                    theme_words = theme.lower().split()
                    for word in theme_words:
                        if len(word) > 3:  # Filter short words
                            theme_count[word] = theme_count.get(word, 0) + 1
                
                # Count styles
                style = content.get("style") or content.get("tone", "")
                if style:
                    style_count[style] = style_count.get(style, 0) + 1
            
            # Store trends analysis
            trends_analysis = {
                "trending_themes": sorted(theme_count.items(), key=lambda x: x[1], reverse=True)[:10],
                "popular_styles": sorted(style_count.items(), key=lambda x: x[1], reverse=True)[:5],
                "analysis_date": datetime.now().isoformat()
            }
            
            await self.store_memory(
                content=trends_analysis,
                memory_type="creative_trends",
                importance=2.0,
                tags=["trends", "analysis", "creative_patterns"]
            )
            
        except Exception as e:
            logger.error(f"Creative trends analysis failed: {e}")
    
    async def _optimize_character_library(self):
        """Optimize character library by removing unused characters."""
        try:
            # Keep only characters that have been referenced recently
            cutoff_date = datetime.now() - timedelta(days=60)
            
            characters_to_remove = []
            for char_id, character in self.character_library.items():
                if character.created_at < cutoff_date:
                    # Check if character has been used in recent works
                    char_used = False
                    recent_works = await self.retrieve_memory(
                        tags=[character.name.lower().replace(" ", "_")],
                        limit=5
                    )
                    
                    if not recent_works:
                        characters_to_remove.append(char_id)
            
            # Remove unused characters
            for char_id in characters_to_remove[:10]:  # Limit removal
                if char_id in self.character_library:
                    del self.character_library[char_id]
                    self.writing_stats["character_count"] -= 1
            
            if characters_to_remove:
                logger.debug(f"Optimized character library: removed {len(characters_to_remove)} unused characters")
                
        except Exception as e:
            logger.error(f"Character library optimization failed: {e}")
    
    async def _agent_shutdown(self):
        """Creative agent specific shutdown tasks."""
        try:
            # Save active projects to persistent storage
            if self.active_projects:
                projects_backup = {
                    project_id: {
                        "title": project.title,
                        "content_type": project.content_type.value,
                        "genre": project.genre.value if project.genre else None,
                        "status": project.status,
                        "created_at": project.created_at.isoformat(),
                        "last_modified": project.last_modified.isoformat(),
                        "metadata": project.metadata,
                        "content_preview": project.content[:500] if project.content else "",
                        "word_count": len(project.content.split()) if project.content else 0
                    }
                    for project_id, project in self.active_projects.items()
                }
                
                await self.store_memory(
                    content=projects_backup,
                    memory_type="projects_backup",
                    importance=4.0,
                    tags=["projects", "backup", "shutdown"]
                )
            
            # Save character library
            if self.character_library:
                characters_backup = {
                    char_id: {
                        "name": char.name,
                        "age": char.age,
                        "occupation": char.occupation,
                        "personality": char.personality,
                        "background": char.background,
                        "goals": char.goals,
                        "created_at": char.created_at.isoformat()
                    }
                    for char_id, char in self.character_library.items()
                }
                
                await self.store_memory(
                    content=characters_backup,
                    memory_type="characters_backup",
                    importance=3.5,
                    tags=["characters", "backup", "shutdown"]
                )
            
            # Save writing statistics
            final_stats = {
                **self.writing_stats,
                "shutdown_time": datetime.now().isoformat(),
                "agent_runtime": (datetime.now() - self.created_at).total_seconds(),
                "active_projects_count": len(self.active_projects),
                "character_library_size": len(self.character_library)
            }
            
            await self.store_memory(
                content=final_stats,
                memory_type="writing_statistics",
                importance=3.0,
                tags=["statistics", "performance", "shutdown"]
            )
            
            logger.info("Creative agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Creative agent shutdown error: {e}")
    
    # Additional utility methods
    
    def get_project_summary(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific project."""
        if project_id not in self.active_projects:
            return None
        
        project = self.active_projects[project_id]
        return {
            "project_id": project_id,
            "title": project.title,
            "content_type": project.content_type.value,
            "genre": project.genre.value if project.genre else None,
            "status": project.status,
            "word_count": len(project.content.split()) if project.content else 0,
            "character_count": len(project.characters),
            "created_at": project.created_at.isoformat(),
            "last_modified": project.last_modified.isoformat()
        }
    
    def get_active_projects(self) -> List[Dict[str, Any]]:
        """Get list of all active projects."""
        return [
            self.get_project_summary(project_id) 
            for project_id in self.active_projects.keys()
        ]
    
    def get_character_summary(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific character."""
        if character_id not in self.character_library:
            return None
        
        character = self.character_library[character_id]
        return {
            "character_id": character_id,
            "name": character.name,
            "age": character.age,
            "occupation": character.occupation,
            "personality_traits": len(character.personality),
            "goals_count": len(character.goals),
            "relationships_count": len(character.relationships),
            "created_at": character.created_at.isoformat()
        }
    
    def get_character_library(self) -> List[Dict[str, Any]]:
        """Get list of all characters in library."""
        return [
            self.get_character_summary(char_id) 
            for char_id in self.character_library.keys()
        ]
    
    def get_writing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive writing statistics."""
        return {
            **self.writing_stats,
            "active_projects": len(self.active_projects),
            "character_library_size": len(self.character_library),
            "agent_uptime": (datetime.now() - self.created_at).total_seconds(),
            "last_updated": datetime.now().isoformat()
        }
    
    async def search_creative_works(self, query: str, work_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search creative works by content."""
        try:
            # Search memory for relevant creative works
            search_tags = [query.replace(" ", "_")]
            if work_types:
                search_tags.extend(work_types)
            
            relevant_works = []
            
            # Search different memory types
            memory_types = ["creative_work", "story_continuation", "character_development", "dialogue"]
            
            for memory_type in memory_types:
                if work_types and memory_type not in work_types:
                    continue
                
                memories = await self.retrieve_memory(
                    memory_type=memory_type,
                    tags=search_tags,
                    limit=10
                )
                
                for memory in memories:
                    content_str = str(memory.content).lower()
                    query_words = set(query.lower().split())
                    content_words = set(content_str.split())
                    
                    # Calculate relevance
                    overlap = len(query_words.intersection(content_words))
                    relevance = overlap / len(query_words) if query_words else 0
                    
                    if relevance > 0.1:
                        relevant_works.append({
                            "memory_id": memory.memory_id,
                            "memory_type": memory.memory_type,
                            "relevance_score": relevance,
                            "content_preview": content_str[:200] + "...",
                            "tags": memory.tags,
                            "created_at": memory.created_at.isoformat()
                        })
            
            # Sort by relevance
            relevant_works.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return relevant_works[:15]  # Return top 15 results
            
        except Exception as e:
            logger.error(f"Creative works search failed: {e}")
            return []
    
    async def export_creative_data(self, format: str = "json") -> Dict[str, Any]:
        """Export creative data in specified format."""
        try:
            export_data = {
                "agent_info": {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "created_at": self.created_at.isoformat(),
                    "capabilities": list(self.capabilities),
                    "creativity_level": self.creativity_level,
                    "style_consistency": self.style_consistency
                },
                "active_projects": {
                    project_id: {
                        "title": project.title,
                        "content_type": project.content_type.value,
                        "genre": project.genre.value if project.genre else None,
                        "status": project.status,
                        "metadata": project.metadata,
                        "created_at": project.created_at.isoformat(),
                        "word_count": len(project.content.split()) if project.content else 0
                    }
                    for project_id, project in list(self.active_projects.items())[:20]  # Limit for export
                },
                "character_library": {
                    char_id: {
                        "name": char.name,
                        "age": char.age,
                        "occupation": char.occupation,
                        "personality": char.personality,
                        "background": char.background[:200] if char.background else "",  # Truncate for export
                        "goals": char.goals,
                        "created_at": char.created_at.isoformat()
                    }
                    for char_id, char in list(self.character_library.items())[:50]  # Limit for export
                },
                "writing_statistics": self.writing_stats,
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
            logger.error(f"Creative data export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_inspiration(self, topic: str = "", style: str = "") -> Dict[str, Any]:
        """Get creative inspiration based on topic or style."""
        try:
            # Retrieve relevant templates and past works for inspiration
            inspiration_sources = []
            
            # Get templates
            templates = await self.retrieve_memory(memory_type="creative_templates", limit=3)
            if templates:
                inspiration_sources.extend([
                    {
                        "type": "template",
                        "content": template.content,
                        "relevance": "structural guidance"
                    }
                    for template in templates
                ])
            
            # Get similar past works
            if topic:
                similar_works = await self.search_creative_works(topic, ["creative_work"])
                inspiration_sources.extend([
                    {
                        "type": "past_work",
                        "content": work["content_preview"],
                        "relevance": f"similar topic (score: {work['relevance_score']:.2f})"
                    }
                    for work in similar_works[:3]
                ])
            
            # Generate inspiration prompt
            inspiration_prompt = f"""Provide creative inspiration for writing about "{topic}" in {style} style.

Include:
1. Unique angles or perspectives to explore
2. Interesting character concepts
3. Potential conflicts or tensions
4. Atmospheric details and settings
5. Themes worth exploring
6. Creative techniques to try

Make the suggestions specific and actionable."""
            
            # Generate inspiration
            if self.model_manager:
                response = await self.model_manager.generate_completion(
                    prompt=inspiration_prompt,
                    temperature=0.8,  # High creativity for inspiration
                    max_tokens=600
                )
                
                if response.success:
                    inspiration_sources.append({
                        "type": "generated_inspiration",
                        "content": response.content,
                        "relevance": "AI-generated suggestions"
                    })
            
            result = {
                "topic": topic,
                "style": style,
                "inspiration_sources": inspiration_sources,
                "source_count": len(inspiration_sources),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Inspiration generation failed: {e}")
            return {
                "topic": topic,
                "style": style,
                "inspiration_sources": [],
                "error": str(e)
            }
