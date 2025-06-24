#!/usr/bin/env python3
"""
User Feedback Manager for Adaptive Learning
Handles user preferences, feedback collection, and taste modeling

Features:
- Feedback collection and storage
- User preference learning
- Context-aware recommendations
- Taste profile modeling
- A/B testing for improvements
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FeedbackEntry:
    """Individual feedback entry."""
    session_id: str
    feedback_type: str  # 'like', 'dislike', 'rating', 'comment', 'accept', 'reject'
    target: str  # 'analysis', 'generation', 'arrangement', 'plugin_result'
    target_id: Optional[str]  # Specific plugin or result ID
    value: Any  # Boolean, rating, or text
    context: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None


@dataclass
class UserProfile:
    """User preference profile."""
    user_id: str
    preferred_genres: List[str]
    preferred_energy_levels: List[float]
    preferred_complexity: List[float]
    plugin_preferences: Dict[str, float]  # Plugin name -> preference score
    feedback_count: int
    last_updated: datetime
    tags: List[str]


@dataclass
class ContextPattern:
    """Discovered pattern in user behavior."""
    pattern_id: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    usage_count: int
    last_seen: datetime


class UserFeedbackManager:
    """
    Manages user feedback collection and learning for personalized recommendations.
    """
    
    def __init__(self, db_path: str = "data/user_feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory caches
        self.user_profiles: Dict[str, UserProfile] = {}
        self.context_patterns: Dict[str, ContextPattern] = {}
        self.feedback_cache: List[FeedbackEntry] = []
        
        # Learning parameters
        self.min_feedback_for_profile = 5
        self.pattern_confidence_threshold = 0.7
        self.cache_update_interval = 300  # 5 minutes
        
        # Feature extractors
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self._load_profiles()
        self._load_patterns()

    def _init_database(self) -> None:
        """Initialize SQLite database for feedback storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    target_id TEXT,
                    value TEXT NOT NULL,
                    context TEXT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Context patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_data TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)')
            
            conn.commit()

    async def log_feedback(
        self,
        session_id: str,
        feedback_type: str,
        target: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        target_id: Optional[str] = None
    ) -> None:
        """Log user feedback entry."""
        
        feedback_entry = FeedbackEntry(
            session_id=session_id,
            feedback_type=feedback_type,
            target=target,
            target_id=target_id,
            value=value,
            context=context or {},
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        # Store in database
        await self._store_feedback(feedback_entry)
        
        # Add to cache
        self.feedback_cache.append(feedback_entry)
        
        # Update user profile if user_id provided
        if user_id:
            await self._update_user_profile(user_id, feedback_entry)
        
        # Learn patterns
        await self._learn_from_feedback(feedback_entry)
        
        self.logger.info(f"Logged {feedback_type} feedback for {target} in session {session_id}")

    async def _store_feedback(self, feedback: FeedbackEntry) -> None:
        """Store feedback entry in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback 
                (session_id, feedback_type, target, target_id, value, context, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.session_id,
                feedback.feedback_type,
                feedback.target,
                feedback.target_id,
                json.dumps(feedback.value),
                json.dumps(feedback.context),
                feedback.timestamp.isoformat(),
                feedback.user_id
            ))
            
            conn.commit()

    async def _update_user_profile(self, user_id: str, feedback: FeedbackEntry) -> None:
        """Update user profile based on feedback."""
        
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferred_genres=[],
                preferred_energy_levels=[],
                preferred_complexity=[],
                plugin_preferences={},
                feedback_count=0,
                last_updated=datetime.now(),
                tags=[]
            )
        
        profile = self.user_profiles[user_id]
        profile.feedback_count += 1
        profile.last_updated = datetime.now()
        
        # Extract preferences from feedback context
        context = feedback.context
        
        # Update genre preferences
        if 'genre' in context and feedback.feedback_type in ['like', 'accept']:
            genre = context['genre']
            if genre not in profile.preferred_genres:
                profile.preferred_genres.append(genre)
        
        # Update energy level preferences
        if 'energy_level' in context and isinstance(feedback.value, (int, float)):
            if feedback.feedback_type in ['like', 'accept'] or (
                feedback.feedback_type == 'rating' and feedback.value >= 4
            ):
                profile.preferred_energy_levels.append(context['energy_level'])
        
        # Update complexity preferences
        if 'complexity' in context and isinstance(feedback.value, (int, float)):
            if feedback.feedback_type in ['like', 'accept'] or (
                feedback.feedback_type == 'rating' and feedback.value >= 4
            ):
                profile.preferred_complexity.append(context['complexity'])
        
        # Update plugin preferences
        if feedback.target == 'generation' and 'plugin_name' in context:
            plugin_name = context['plugin_name']
            
            # Calculate preference score based on feedback
            if feedback.feedback_type == 'like' or feedback.feedback_type == 'accept':
                score_delta = 0.1
            elif feedback.feedback_type == 'dislike' or feedback.feedback_type == 'reject':
                score_delta = -0.1
            elif feedback.feedback_type == 'rating':
                # Convert rating (1-5) to score delta (-0.2 to +0.2)
                score_delta = (feedback.value - 3) * 0.1
            else:
                score_delta = 0
            
            current_score = profile.plugin_preferences.get(plugin_name, 0.0)
            profile.plugin_preferences[plugin_name] = max(-1.0, min(1.0, current_score + score_delta))
        
        # Store updated profile
        await self._store_user_profile(profile)

    async def _store_user_profile(self, profile: UserProfile) -> None:
        """Store user profile in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles (user_id, profile_data, last_updated)
                VALUES (?, ?, ?)
            ''', (
                profile.user_id,
                json.dumps(asdict(profile), default=str),
                profile.last_updated.isoformat()
            ))
            
            conn.commit()

    async def _learn_from_feedback(self, feedback: FeedbackEntry) -> None:
        """Learn patterns from feedback."""
        
        # Extract context features
        context_features = self._extract_context_features(feedback.context)
        
        # Find similar contexts in history
        similar_contexts = await self._find_similar_contexts(context_features)
        
        # Update or create patterns
        for similar_context in similar_contexts:
            pattern_id = self._generate_pattern_id(similar_context)
            
            if pattern_id in self.context_patterns:
                pattern = self.context_patterns[pattern_id]
                pattern.usage_count += 1
                pattern.last_seen = datetime.now()
                
                # Update outcomes based on feedback
                self._update_pattern_outcomes(pattern, feedback)
            else:
                # Create new pattern
                self.context_patterns[pattern_id] = ContextPattern(
                    pattern_id=pattern_id,
                    conditions=similar_context,
                    outcomes=self._extract_outcomes(feedback),
                    confidence=0.5,
                    usage_count=1,
                    last_seen=datetime.now()
                )

    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from context."""
        features = {}
        
        # Musical features
        if 'genre' in context:
            features['genre'] = context['genre']
        if 'tempo' in context:
            features['tempo_range'] = self._categorize_tempo(context['tempo'])
        if 'key' in context:
            features['key_mode'] = self._extract_key_mode(context['key'])
        if 'energy_level' in context:
            features['energy_category'] = self._categorize_energy(context['energy_level'])
        if 'complexity' in context:
            features['complexity_level'] = self._categorize_complexity(context['complexity'])
        
        # Plugin features
        if 'plugin_name' in context:
            features['plugin_type'] = self._categorize_plugin(context['plugin_name'])
        
        # Time features
        now = datetime.now()
        features['time_of_day'] = self._categorize_time_of_day(now)
        features['day_of_week'] = now.strftime('%A')
        
        return features

    def _categorize_tempo(self, tempo: float) -> str:
        """Categorize tempo into ranges."""
        if tempo < 80:
            return 'slow'
        elif tempo < 120:
            return 'medium'
        elif tempo < 160:
            return 'fast'
        else:
            return 'very_fast'

    def _extract_key_mode(self, key_string: str) -> str:
        """Extract mode from key string."""
        if 'minor' in key_string.lower():
            return 'minor'
        else:
            return 'major'

    def _categorize_energy(self, energy: float) -> str:
        """Categorize energy level."""
        if energy < 0.3:
            return 'low'
        elif energy < 0.7:
            return 'medium'
        else:
            return 'high'

    def _categorize_complexity(self, complexity: float) -> str:
        """Categorize complexity level."""
        if complexity < 0.3:
            return 'simple'
        elif complexity < 0.7:
            return 'moderate'
        else:
            return 'complex'

    def _categorize_plugin(self, plugin_name: str) -> str:
        """Categorize plugin by type."""
        if 'drum' in plugin_name.lower():
            return 'drums'
        elif 'bass' in plugin_name.lower():
            return 'bass'
        elif 'guitar' in plugin_name.lower():
            return 'guitar'
        elif 'piano' in plugin_name.lower() or 'keys' in plugin_name.lower():
            return 'keyboard'
        else:
            return 'other'

    def _categorize_time_of_day(self, dt: datetime) -> str:
        """Categorize time of day."""
        hour = dt.hour
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    async def _find_similar_contexts(
        self, 
        context_features: Dict[str, Any], 
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar contexts from history."""
        
        # Get recent feedback for comparison
        recent_feedback = await self._get_recent_feedback(days=30)
        
        similar_contexts = []
        
        for feedback in recent_feedback:
            other_features = self._extract_context_features(feedback.context)
            
            # Calculate similarity
            similarity = self._calculate_context_similarity(context_features, other_features)
            
            if similarity >= threshold:
                similar_contexts.append(other_features)
        
        return similar_contexts

    def _calculate_context_similarity(
        self, 
        context1: Dict[str, Any], 
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two contexts."""
        
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
        
        return matches / len(common_keys)

    def _generate_pattern_id(self, context: Dict[str, Any]) -> str:
        """Generate unique pattern ID from context."""
        # Sort keys for consistent ID generation
        sorted_items = sorted(context.items())
        pattern_str = "_".join([f"{k}:{v}" for k, v in sorted_items])
        return f"pattern_{hash(pattern_str) % 1000000}"

    def _extract_outcomes(self, feedback: FeedbackEntry) -> Dict[str, Any]:
        """Extract outcomes from feedback."""
        outcomes = {
            'feedback_type': feedback.feedback_type,
            'target': feedback.target,
            'positive': feedback.feedback_type in ['like', 'accept'] or (
                feedback.feedback_type == 'rating' and 
                isinstance(feedback.value, (int, float)) and 
                feedback.value >= 4
            )
        }
        
        if feedback.target_id:
            outcomes['target_id'] = feedback.target_id
        
        return outcomes

    def _update_pattern_outcomes(self, pattern: ContextPattern, feedback: FeedbackEntry) -> None:
        """Update pattern outcomes based on new feedback."""
        outcomes = self._extract_outcomes(feedback)
        
        # Simple reinforcement learning approach
        for key, value in outcomes.items():
            if key in pattern.outcomes:
                # Weighted average with new feedback
                weight = 0.1  # Learning rate
                if isinstance(pattern.outcomes[key], (int, float)) and isinstance(value, (int, float)):
                    pattern.outcomes[key] = (1 - weight) * pattern.outcomes[key] + weight * value
                else:
                    pattern.outcomes[key] = value
            else:
                pattern.outcomes[key] = value
        
        # Update confidence based on usage
        pattern.confidence = min(1.0, pattern.confidence + 0.05)

    async def get_user_context(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user context for personalized recommendations."""
        
        context = {
            'session_id': session_id,
            'has_profile': False,
            'recommendations': {}
        }
        
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            context.update({
                'has_profile': True,
                'preferred_genres': profile.preferred_genres,
                'plugin_preferences': profile.plugin_preferences,
                'feedback_count': profile.feedback_count
            })
            
            # Add personalized recommendations
            context['recommendations'] = await self._generate_recommendations(profile)
        
        # Add session-specific patterns
        session_patterns = await self._get_session_patterns(session_id)
        if session_patterns:
            context['session_patterns'] = session_patterns
        
        return context

    async def _generate_recommendations(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate personalized recommendations based on user profile."""
        
        recommendations = {}
        
        # Plugin recommendations
        if profile.plugin_preferences:
            sorted_plugins = sorted(
                profile.plugin_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )
            recommendations['preferred_plugins'] = [name for name, score in sorted_plugins if score > 0]
        
        # Parameter recommendations
        if profile.preferred_energy_levels:
            avg_energy = np.mean(profile.preferred_energy_levels)
            recommendations['suggested_energy'] = float(avg_energy)
        
        if profile.preferred_complexity:
            avg_complexity = np.mean(profile.preferred_complexity)
            recommendations['suggested_complexity'] = float(avg_complexity)
        
        # Genre recommendations
        if profile.preferred_genres:
            genre_counts = Counter(profile.preferred_genres)
            recommendations['preferred_genres'] = [
                genre for genre, count in genre_counts.most_common(3)
            ]
        
        return recommendations

    async def _get_session_patterns(self, session_id: str) -> List[Dict[str, Any]]:
        """Get patterns relevant to current session."""
        
        # Get session feedback
        session_feedback = await self._get_session_feedback_internal(session_id)
        
        if not session_feedback:
            return []
        
        # Extract session context
        session_contexts = [
            self._extract_context_features(fb.context) 
            for fb in session_feedback
        ]
        
        # Find matching patterns
        matching_patterns = []
        
        for pattern in self.context_patterns.values():
            if pattern.confidence >= self.pattern_confidence_threshold:
                for session_context in session_contexts:
                    similarity = self._calculate_context_similarity(
                        pattern.conditions, 
                        session_context
                    )
                    
                    if similarity >= 0.8:
                        matching_patterns.append({
                            'pattern_id': pattern.pattern_id,
                            'conditions': pattern.conditions,
                            'outcomes': pattern.outcomes,
                            'confidence': pattern.confidence,
                            'similarity': similarity
                        })
                        break
        
        # Sort by confidence and similarity
        matching_patterns.sort(
            key=lambda x: (x['confidence'], x['similarity']), 
            reverse=True
        )
        
        return matching_patterns[:5]  # Return top 5 patterns

    async def get_session_feedback(self, session_id: str) -> Dict[str, Any]:
        """Get all feedback for a session."""
        
        feedback_entries = await self._get_session_feedback_internal(session_id)
        
        return {
            'session_id': session_id,
            'feedback_count': len(feedback_entries),
            'feedback_entries': [asdict(fb) for fb in feedback_entries],
            'summary': self._summarize_session_feedback(feedback_entries)
        }

    async def _get_session_feedback_internal(self, session_id: str) -> List[FeedbackEntry]:
        """Get feedback entries for a session."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT session_id, feedback_type, target, target_id, value, context, timestamp, user_id
                FROM feedback
                WHERE session_id = ?
                ORDER BY timestamp
            ''', (session_id,))
            
            rows = cursor.fetchall()
            
            feedback_entries = []
            for row in rows:
                feedback_entries.append(FeedbackEntry(
                    session_id=row[0],
                    feedback_type=row[1],
                    target=row[2],
                    target_id=row[3],
                    value=json.loads(row[4]),
                    context=json.loads(row[5]) if row[5] else {},
                    timestamp=datetime.fromisoformat(row[6]),
                    user_id=row[7]
                ))
            
            return feedback_entries

    def _summarize_session_feedback(self, feedback_entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Summarize feedback for a session."""
        
        if not feedback_entries:
            return {}
        
        summary = {
            'total_feedback': len(feedback_entries),
            'feedback_types': Counter([fb.feedback_type for fb in feedback_entries]),
            'targets': Counter([fb.target for fb in feedback_entries]),
            'positive_feedback': 0,
            'negative_feedback': 0,
            'average_rating': None
        }
        
        ratings = []
        
        for fb in feedback_entries:
            if fb.feedback_type in ['like', 'accept']:
                summary['positive_feedback'] += 1
            elif fb.feedback_type in ['dislike', 'reject']:
                summary['negative_feedback'] += 1
            elif fb.feedback_type == 'rating' and isinstance(fb.value, (int, float)):
                ratings.append(fb.value)
                if fb.value >= 4:
                    summary['positive_feedback'] += 1
                elif fb.value <= 2:
                    summary['negative_feedback'] += 1
        
        if ratings:
            summary['average_rating'] = float(np.mean(ratings))
        
        return summary

    async def _get_recent_feedback(self, days: int = 30) -> List[FeedbackEntry]:
        """Get recent feedback entries."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT session_id, feedback_type, target, target_id, value, context, timestamp, user_id
                FROM feedback
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (cutoff_date.isoformat(),))
            
            rows = cursor.fetchall()
            
            feedback_entries = []
            for row in rows:
                feedback_entries.append(FeedbackEntry(
                    session_id=row[0],
                    feedback_type=row[1],
                    target=row[2],
                    target_id=row[3],
                    value=json.loads(row[4]),
                    context=json.loads(row[5]) if row[5] else {},
                    timestamp=datetime.fromisoformat(row[6]),
                    user_id=row[7]
                ))
            
            return feedback_entries

    def _load_profiles(self) -> None:
        """Load user profiles from database."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT user_id, profile_data, last_updated
                    FROM user_profiles
                ''')
                
                rows = cursor.fetchall()
                
                for row in rows:
                    profile_data = json.loads(row[1])
                    
                    # Handle datetime parsing
                    for date_field in ['last_updated']:
                        if date_field in profile_data:
                            profile_data[date_field] = datetime.fromisoformat(profile_data[date_field])
                    
                    profile = UserProfile(**profile_data)
                    self.user_profiles[profile.user_id] = profile
                
                self.logger.info(f"Loaded {len(self.user_profiles)} user profiles")
                
        except Exception as e:
            self.logger.error(f"Failed to load user profiles: {e}")

    def _load_patterns(self) -> None:
        """Load context patterns from database."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT pattern_id, pattern_data, last_updated
                    FROM context_patterns
                ''')
                
                rows = cursor.fetchall()
                
                for row in rows:
                    pattern_data = json.loads(row[1])
                    
                    # Handle datetime parsing
                    for date_field in ['last_seen']:
                        if date_field in pattern_data:
                            pattern_data[date_field] = datetime.fromisoformat(pattern_data[date_field])
                    
                    pattern = ContextPattern(**pattern_data)
                    self.context_patterns[pattern.pattern_id] = pattern
                
                self.logger.info(f"Loaded {len(self.context_patterns)} context patterns")
                
        except Exception as e:
            self.logger.error(f"Failed to load context patterns: {e}")

    async def log_analysis_session(
        self,
        session_id: str,
        context: Any,  # AnalysisContext
        results: Dict[str, Any]
    ) -> None:
        """Log an analysis session for learning."""
        
        # Extract useful context for learning
        analysis_context = {
            'input_file': getattr(context, 'input_path', None),
            'file_type': getattr(context, 'file_info', {}).get('file_type'),
            'plugins_executed': results.get('plugins_executed', []),
            'execution_time': results.get('execution_time', 0),
            'missing_instruments_found': len(results.get('missing_instruments', []))
        }
        
        # Auto-feedback for successful analysis
        await self.log_feedback(
            session_id=session_id,
            feedback_type='analysis_complete',
            target='analysis',
            value=True,
            context=analysis_context
        )

    async def log_generation(
        self,
        session_id: str,
        role: str,
        plugin_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Log a generation event for learning."""
        
        generation_context = {
            'role': role,
            'plugin_name': plugin_name,
            'parameters': parameters,
            'generation_successful': result.get('status') == 'success',
            'generation_time': result.get('execution_time', 0)
        }
        
        # Auto-feedback for successful generation
        await self.log_feedback(
            session_id=session_id,
            feedback_type='generation_complete',
            target='generation',
            target_id=f"{plugin_name}_{role}",
            value=result.get('status') == 'success',
            context=generation_context
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback system statistics."""
        
        return {
            'user_profiles': len(self.user_profiles),
            'context_patterns': len(self.context_patterns),
            'cached_feedback': len(self.feedback_cache),
            'total_feedback_in_db': self._get_total_feedback_count(),
            'pattern_confidence_threshold': self.pattern_confidence_threshold,
            'min_feedback_for_profile': self.min_feedback_for_profile
        }

    def _get_total_feedback_count(self) -> int:
        """Get total feedback count from database."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM feedback')
                return cursor.fetchone()[0]
        except:
            return 0


# Factory function
def create_user_feedback_manager(db_path: str = "data/user_feedback.db") -> UserFeedbackManager:
    """Create and return a user feedback manager instance."""
    return UserFeedbackManager(db_path)


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="User Feedback Manager")
    parser.add_argument("command", choices=["stats", "test", "export"])
    parser.add_argument("--db-path", default="data/user_feedback.db", help="Database path")
    
    args = parser.parse_args()
    
    async def main():
        manager = create_user_feedback_manager(args.db_path)
        
        if args.command == "stats":
            stats = manager.get_stats()
            print(json.dumps(stats, indent=2))
        
        elif args.command == "test":
            # Test feedback logging
            await manager.log_feedback(
                session_id="test_session",
                feedback_type="like",
                target="generation",
                value=True,
                context={"plugin_name": "drummaroo", "genre": "rock"},
                user_id="test_user"
            )
            print("Test feedback logged successfully")
        
        elif args.command == "export":
            # Export all data
            recent_feedback = await manager._get_recent_feedback(days=365)
            export_data = {
                'feedback_entries': [asdict(fb) for fb in recent_feedback],
                'user_profiles': {uid: asdict(profile) for uid, profile in manager.user_profiles.items()},
                'patterns': {pid: asdict(pattern) for pid, pattern in manager.context_patterns.items()}
            }
            
            with open("feedback_export.json", "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print("Data exported to feedback_export.json")
    
    asyncio.run(main())