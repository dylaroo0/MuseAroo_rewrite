# 💾 **MuseAroo Database Schema & Data Architecture**
## *Intelligent Data Design for Real-Time Musical Intelligence*

---

## **🔥 DATABASE VISION**

**MuseAroo's database isn't just storage** - it's a **living musical intelligence system** that preserves the complete creative journey, learns from every interaction, and provides lightning-fast access to musical context that enables sub-150ms AI generation.

### **🎯 Core Data Principles:**
- **⚡ Sub-Millisecond Query Response** - Critical data accessible in <1ms for real-time generation
- **🧠 Musical Intelligence Storage** - Sophisticated indexing for musical patterns and relationships
- **🔄 Real-Time Session State** - Live collaborative sessions with instant synchronization
- **📈 Learning System Integration** - Every interaction feeds the AI learning pipeline
- **🌍 Global Distribution** - Data replicated to edge locations for minimum latency
- **🛡️ Creative IP Protection** - Secure storage with granular access control

---

## **🏗️ DATABASE ARCHITECTURE**

### **🎯 Multi-Database Strategy:**
```
┌─────────────────────────────────────────────────────┐
│                POSTGRESQL CLUSTER                   │
│               (Primary Relational)                  │
├─────────────────┬─────────────────┬─────────────────┤
│   User Data     │   Projects      │   Generations   │
│   Sessions      │   Preferences   │   Collaborations│
└─────────────────┴─────────────────┴─────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│              TIMESCALEDB EXTENSION                  │
│             (Time-Series Audio Data)                │
├─────────────────┬─────────────────┬─────────────────┤
│ Audio Analysis  │ Performance     │ User Behavior   │
│ Timing Metadata │ Metrics         │ Learning Data   │
└─────────────────┴─────────────────┴─────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                  REDIS CLUSTER                      │
│              (Real-Time Cache/State)                │
├─────────────────┬─────────────────┬─────────────────┤
│ Session State   │ Generated       │ WebSocket       │
│ User Presence   │ Pattern Cache   │ Message Queue   │
└─────────────────┴─────────────────┴─────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│               ELASTICSEARCH                         │
│            (Musical Pattern Search)                 │
├─────────────────┬─────────────────┬─────────────────┤
│ Pattern Index   │ Style Search    │ Similarity      │
│ Semantic Search │ Cultural Tags   │ Matching        │
└─────────────────┴─────────────────┴─────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                  S3 STORAGE                         │
│               (Audio/MIDI Files)                    │
├─────────────────┬─────────────────┬─────────────────┤
│ Generated Audio │ User Uploads    │ Exported        │
│ MIDI Files      │ Project Files   │ Sessions        │
└─────────────────┴─────────────────┴─────────────────┘
```

### **🎵 Core Database Technologies:**
- **PostgreSQL 15+** with TimescaleDB for relational and time-series data
- **Redis Cluster** for real-time state and high-speed caching
- **Elasticsearch** for musical pattern search and similarity matching
- **S3-Compatible Storage** with global CDN for audio/MIDI files
- **PgVector Extension** for AI embedding storage and similarity search

---

## **🎼 CORE SCHEMA DESIGN**

### **👥 User & Authentication Schema:**
```sql
-- Users table with comprehensive profile data
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Profile information
    display_name VARCHAR(100),
    bio TEXT,
    profile_image_url VARCHAR(500),
    
    -- Musical profile
    primary_instruments TEXT[], -- Array of instruments
    musical_styles TEXT[],      -- Preferred genres/styles
    skill_level user_skill_level DEFAULT 'beginner',
    
    -- Subscription and permissions
    subscription_tier subscription_tier DEFAULT 'free',
    subscription_expires_at TIMESTAMPTZ,
    total_generations INTEGER DEFAULT 0,
    monthly_generations INTEGER DEFAULT 0,
    
    -- Accessibility preferences
    accessibility_settings JSONB DEFAULT '{}',
    voice_enabled BOOLEAN DEFAULT true,
    ui_preferences JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT username_length CHECK (char_length(username) >= 3)
);

-- Custom enums for type safety
CREATE TYPE user_skill_level AS ENUM ('beginner', 'intermediate', 'advanced', 'professional');
CREATE TYPE subscription_tier AS ENUM ('free', 'pro', 'studio', 'enterprise');

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_subscription ON users(subscription_tier, subscription_expires_at);
CREATE INDEX idx_users_last_login ON users(last_login_at);

-- User sessions for authentication
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Index for fast session lookup
    INDEX idx_sessions_token (session_token),
    INDEX idx_sessions_user (user_id),
    INDEX idx_sessions_expires (expires_at)
);
```

### **🎨 Projects & Creative Sessions:**
```sql
-- Projects: Top-level creative containers
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Project metadata
    name VARCHAR(200) NOT NULL,
    description TEXT,
    project_image_url VARCHAR(500),
    
    -- Musical metadata
    original_audio_url VARCHAR(500),
    original_audio_analysis JSONB, -- BrainAroo analysis results
    musical_key VARCHAR(10),
    tempo DECIMAL(5,2),
    time_signature VARCHAR(10),
    style_tags TEXT[],
    
    -- Collaboration
    collaborators UUID[], -- Array of user IDs
    is_public BOOLEAN DEFAULT false,
    collaboration_settings JSONB DEFAULT '{}',
    
    -- Status and organization
    status project_status DEFAULT 'active',
    folder_path VARCHAR(500), -- For project organization
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', name || ' ' || COALESCE(description, ''))
    ) STORED
);

CREATE TYPE project_status AS ENUM ('active', 'archived', 'shared', 'deleted');

-- Indexes for project queries
CREATE INDEX idx_projects_user ON projects(user_id, updated_at DESC);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_search ON projects USING GIN(search_vector);
CREATE INDEX idx_projects_collaborators ON projects USING GIN(collaborators);

-- Generation sessions: Individual AI generation workflows
CREATE TABLE generation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session metadata
    session_name VARCHAR(200),
    session_type session_type NOT NULL,
    
    -- Musical context
    input_audio_url VARCHAR(500),
    musical_context JSONB NOT NULL, -- Complete MusicalContext from ContextManager
    
    -- Generation configuration
    active_engines TEXT[] NOT NULL, -- Which Roo engines are active
    engine_parameters JSONB NOT NULL, -- Parameters for each engine
    generation_goals TEXT[], -- User's stated goals for this session
    
    -- Session state
    current_step generation_step DEFAULT 'analysis',
    completion_percentage DECIMAL(5,2) DEFAULT 0,
    session_state JSONB DEFAULT '{}',
    
    -- Results
    generated_content_urls TEXT[], -- Array of generated audio/MIDI URLs
    final_export_url VARCHAR(500),
    user_satisfaction_score DECIMAL(3,2), -- 0-5 rating
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TYPE session_type AS ENUM ('drum_generation', 'full_arrangement', 'style_exploration', 'collaboration');
CREATE TYPE generation_step AS ENUM ('analysis', 'generation', 'refinement', 'arrangement', 'mixing', 'export');

-- Indexes for session queries
CREATE INDEX idx_sessions_project ON generation_sessions(project_id, created_at DESC);
CREATE INDEX idx_sessions_user ON generation_sessions(user_id, created_at DESC);
CREATE INDEX idx_sessions_type ON generation_sessions(session_type);
```

### **🎵 Musical Intelligence & Pattern Storage:**
```sql
-- Generated patterns: Individual AI-generated musical elements
CREATE TABLE generated_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES generation_sessions(id) ON DELETE CASCADE,
    
    -- Pattern metadata
    engine_name VARCHAR(50) NOT NULL, -- drummaroo, bassaroo, etc.
    pattern_type pattern_type NOT NULL,
    pattern_name VARCHAR(200),
    
    -- Musical characteristics
    musical_key VARCHAR(10),
    tempo DECIMAL(5,2),
    time_signature VARCHAR(10),
    duration_ms INTEGER,
    complexity_score DECIMAL(3,2), -- 0-1 complexity rating
    
    -- Generation details
    generation_parameters JSONB NOT NULL,
    generation_algorithm VARCHAR(100),
    confidence_score DECIMAL(3,2), -- AI confidence in result
    generation_time_ms INTEGER,
    
    -- File references
    midi_file_url VARCHAR(500),
    audio_file_url VARCHAR(500),
    notation_file_url VARCHAR(500),
    
    -- Pattern data for analysis
    midi_data BYTEA, -- Compressed MIDI data
    pattern_features VECTOR(512), -- AI embeddings for similarity search
    audio_fingerprint BYTEA, -- Audio fingerprint for duplicate detection
    
    -- User feedback
    user_rating DECIMAL(3,2), -- 0-5 user rating
    user_feedback TEXT,
    usage_count INTEGER DEFAULT 0, -- How often this pattern is used
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TYPE pattern_type AS ENUM ('drum_pattern', 'bass_line', 'chord_progression', 'melody', 'arrangement', 'mix');

-- Indexes for pattern queries and similarity search
CREATE INDEX idx_patterns_session ON generated_patterns(session_id);
CREATE INDEX idx_patterns_engine ON generated_patterns(engine_name, pattern_type);
CREATE INDEX idx_patterns_musical ON generated_patterns(musical_key, tempo);
CREATE INDEX idx_patterns_quality ON generated_patterns(confidence_score DESC, user_rating DESC);

-- Vector similarity index for pattern matching
CREATE INDEX idx_patterns_similarity ON generated_patterns USING ivfflat (pattern_features vector_cosine_ops);

-- Musical style analysis: Cultural and stylistic tagging
CREATE TABLE musical_styles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    parent_style_id UUID REFERENCES musical_styles(id),
    
    -- Style characteristics
    description TEXT,
    cultural_origin VARCHAR(100),
    historical_period VARCHAR(100),
    typical_instruments TEXT[],
    rhythmic_characteristics JSONB,
    harmonic_characteristics JSONB,
    melodic_characteristics JSONB,
    
    -- AI embeddings for style matching
    style_features VECTOR(256),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pattern-style relationships (many-to-many)
CREATE TABLE pattern_styles (
    pattern_id UUID REFERENCES generated_patterns(id) ON DELETE CASCADE,
    style_id UUID REFERENCES musical_styles(id) ON DELETE CASCADE,
    confidence DECIMAL(3,2) NOT NULL, -- How strongly this pattern matches this style
    
    PRIMARY KEY (pattern_id, style_id)
);
```

### **⏱️ Time-Series Performance Data:**
```sql
-- Using TimescaleDB extension for time-series data
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Performance metrics: Real-time system performance tracking
CREATE TABLE performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    session_id UUID,
    user_id UUID,
    
    -- Latency metrics
    generation_latency_ms INTEGER,
    network_latency_ms INTEGER,
    total_request_time_ms INTEGER,
    
    -- Quality metrics
    generation_confidence DECIMAL(3,2),
    user_satisfaction DECIMAL(3,2),
    cache_hit_rate DECIMAL(3,2),
    
    -- System metrics
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb INTEGER,
    active_sessions INTEGER,
    queue_length INTEGER,
    
    -- Engine-specific metrics
    engine_name VARCHAR(50),
    engine_parameters JSONB,
    algorithm_used VARCHAR(100),
    
    -- Geographic data
    user_region VARCHAR(50),
    edge_server VARCHAR(50)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('performance_metrics', 'time');

-- Create time-based indexes
CREATE INDEX idx_performance_time ON performance_metrics (time DESC);
CREATE INDEX idx_performance_session ON performance_metrics (session_id, time DESC);
CREATE INDEX idx_performance_engine ON performance_metrics (engine_name, time DESC);

-- Audio analysis data: Time-series storage of audio analysis results
CREATE TABLE audio_analysis_data (
    time TIMESTAMPTZ NOT NULL,
    session_id UUID NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- 'spectral', 'rhythmic', 'harmonic', etc.
    
    -- Analysis results
    analysis_data JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    processing_time_ms INTEGER,
    
    -- Audio segment information
    start_time_ms INTEGER,
    duration_ms INTEGER,
    audio_fingerprint BYTEA
);

SELECT create_hypertable('audio_analysis_data', 'time');

-- User behavior analytics: Track user interaction patterns
CREATE TABLE user_behavior_events (
    time TIMESTAMPTZ NOT NULL,
    user_id UUID NOT NULL,
    session_id UUID,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL, -- 'generation_request', 'parameter_change', 'voice_command', etc.
    event_data JSONB NOT NULL,
    
    -- Context
    page_url VARCHAR(500),
    user_agent TEXT,
    device_type VARCHAR(50),
    
    -- Performance
    response_time_ms INTEGER
);

SELECT create_hypertable('user_behavior_events', 'time');
```

---

## **🚀 REAL-TIME STATE MANAGEMENT**

### **🔄 Redis Schema for Live Sessions:**
```python
# Redis data structures for real-time collaboration

# Active session state (Hash)
# Key: session:{session_id}
HSET session:550e8400-e29b-41d4-a716-446655440000 {
    "user_id": "user:123",
    "project_id": "proj:456", 
    "status": "generating",
    "current_engine": "drummaroo",
    "collaborators": ["user:124", "user:125"],
    "last_activity": "2024-06-19T15:30:00Z",
    "musical_context": "{...compressed_json...}",
    "generation_queue": "3",
    "voice_enabled": "true"
}

# User presence tracking (Sorted Set)
# Key: presence:project:{project_id}
# Score: timestamp, Member: user_id
ZADD presence:project:proj456 1718811000 user:123
ZADD presence:project:proj456 1718811005 user:124

# Generation queue (List)
# Key: gen_queue:{session_id}
LPUSH gen_queue:session123 {
    "engine": "drummaroo",
    "parameters": {...},
    "priority": "high",
    "timestamp": "2024-06-19T15:30:00Z"
}

# WebSocket connections (Hash)
# Key: ws_connections:{user_id}
HSET ws_connections:user123 {
    "connection_id": "conn_abc123",
    "session_id": "session_456",
    "last_ping": "2024-06-19T15:30:00Z",
    "capabilities": ["voice", "realtime_generation"]
}

# Generated pattern cache (Hash with TTL)
# Key: pattern_cache:{pattern_hash}
HSET pattern_cache:sha256abc123 {
    "pattern_data": "{...compressed_midi...}",
    "confidence": "0.95",
    "generation_time": "45ms",
    "created_at": "2024-06-19T15:30:00Z"
}
EXPIRE pattern_cache:sha256abc123 3600  # 1 hour TTL

# User preference cache (Hash)
# Key: user_prefs:{user_id}
HSET user_prefs:user123 {
    "voice_enabled": "true",
    "default_complexity": "0.7",
    "preferred_styles": '["jazz", "folk"]',
    "accessibility_mode": "dyslexia_friendly"
}
```

### **⚡ High-Performance Query Patterns:**
```sql
-- Optimized queries for sub-millisecond response times

-- Get active session with musical context (< 1ms target)
PREPARE get_active_session AS
SELECT 
    gs.id,
    gs.musical_context,
    gs.active_engines,
    gs.engine_parameters,
    p.name as project_name,
    u.accessibility_settings
FROM generation_sessions gs
JOIN projects p ON gs.project_id = p.id
JOIN users u ON gs.user_id = u.id
WHERE gs.id = $1 AND gs.current_step != 'completed';

-- Find similar patterns for cache optimization (< 5ms target)
PREPARE find_similar_patterns AS
SELECT 
    id,
    pattern_features <-> $1 as distance,
    midi_file_url,
    confidence_score
FROM generated_patterns
WHERE 
    engine_name = $2 
    AND pattern_features <-> $1 < 0.3  -- Similarity threshold
ORDER BY pattern_features <-> $1
LIMIT 5;

-- Get user's recent generation history (< 2ms target)
PREPARE get_user_recent_generations AS
SELECT 
    gp.id,
    gp.pattern_name,
    gp.confidence_score,
    gp.user_rating,
    gp.created_at
FROM generated_patterns gp
JOIN generation_sessions gs ON gp.session_id = gs.id
WHERE 
    gs.user_id = $1 
    AND gp.created_at > NOW() - INTERVAL '7 days'
ORDER BY gp.created_at DESC
LIMIT 20;

-- Update session state with optimistic locking
PREPARE update_session_state AS
UPDATE generation_sessions
SET 
    session_state = $2,
    current_step = $3,
    completion_percentage = $4,
    updated_at = NOW()
WHERE 
    id = $1 
    AND updated_at = $5  -- Optimistic lock check
RETURNING updated_at;
```

---

## **🔍 SEARCH & INDEXING STRATEGY**

### **🎵 Elasticsearch Musical Pattern Index:**
```json
{
  "mappings": {
    "properties": {
      "pattern_id": {"type": "keyword"},
      "engine_name": {"type": "keyword"},
      "pattern_type": {"type": "keyword"},
      
      "musical_characteristics": {
        "properties": {
          "key": {"type": "keyword"},
          "tempo": {"type": "float"},
          "time_signature": {"type": "keyword"},
          "complexity": {"type": "float"},
          "style_tags": {"type": "keyword"},
          "cultural_origin": {"type": "keyword"}
        }
      },
      
      "pattern_features": {
        "type": "dense_vector",
        "dims": 512,
        "index": true,
        "similarity": "cosine"
      },
      
      "user_feedback": {
        "properties": {
          "rating": {"type": "float"},
          "feedback_text": {
            "type": "text",
            "analyzer": "musical_analyzer"
          },
          "usage_count": {"type": "integer"}
        }
      },
      
      "generation_metadata": {
        "properties": {
          "generation_time": {"type": "date"},
          "confidence_score": {"type": "float"},
          "algorithm_used": {"type": "keyword"},
          "parameters": {"type": "object", "enabled": false}
        }
      }
    }
  },
  
  "settings": {
    "analysis": {
      "analyzer": {
        "musical_analyzer": {
          "tokenizer": "standard",
          "filter": ["lowercase", "musical_synonyms"]
        }
      },
      "filter": {
        "musical_synonyms": {
          "type": "synonym",
          "synonyms": [
            "drums,percussion,beats",
            "bass,bassline,foundation",
            "melody,tune,hook",
            "harmony,chords,progression"
          ]
        }
      }
    }
  }
}
```

### **🧠 AI Embedding Storage:**
```sql
-- Vector embeddings for AI similarity matching
CREATE TABLE ai_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL, -- 'pattern', 'style', 'user_preference'
    entity_id UUID NOT NULL,
    
    -- Embedding data
    embedding_type VARCHAR(50) NOT NULL, -- 'musical_features', 'style_vector', 'user_taste'
    embedding_vector VECTOR(512) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL, -- Model used to generate embedding
    model_version VARCHAR(20) NOT NULL,
    
    -- Metadata
    confidence_score DECIMAL(3,2),
    generation_context JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure uniqueness per entity and embedding type
    UNIQUE(entity_type, entity_id, embedding_type)
);

-- High-performance vector similarity index
CREATE INDEX idx_embeddings_similarity 
ON ai_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 1000);

-- Index for fast entity lookup
CREATE INDEX idx_embeddings_entity ON ai_embeddings(entity_type, entity_id);

-- User preference embeddings for personalization
CREATE TABLE user_preference_embeddings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    
    -- Preference vectors
    style_preferences VECTOR(256), -- Musical style preferences
    complexity_preferences VECTOR(128), -- Complexity level preferences  
    cultural_preferences VECTOR(128), -- Cultural/regional preferences
    instrument_preferences VECTOR(128), -- Instrument preferences
    
    -- Learning metadata
    confidence_score DECIMAL(3,2),
    learning_sample_count INTEGER,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    -- Embedding generation info
    model_version VARCHAR(20),
    generation_context JSONB
);

-- Vector similarity index for user matching
CREATE INDEX idx_user_prefs_style 
ON user_preference_embeddings 
USING ivfflat (style_preferences vector_cosine_ops);
```

---

## **📊 ANALYTICS & LEARNING DATA**

### **🎯 User Learning & Feedback:**
```sql
-- User feedback and learning data
CREATE TABLE user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pattern_id UUID REFERENCES generated_patterns(id) ON DELETE CASCADE,
    session_id UUID REFERENCES generation_sessions(id) ON DELETE CASCADE,
    
    -- Feedback details
    feedback_type feedback_type NOT NULL,
    rating DECIMAL(3,2), -- 0-5 scale
    feedback_text TEXT,
    
    -- Specific feedback categories
    musical_accuracy DECIMAL(3,2),
    style_appropriateness DECIMAL(3,2), 
    creative_inspiration DECIMAL(3,2),
    technical_quality DECIMAL(3,2),
    
    -- Context
    feedback_context JSONB, -- What user was trying to achieve
    voice_command TEXT, -- Original voice command if applicable
    
    -- Action taken
    user_action VARCHAR(50), -- 'kept', 'regenerated', 'modified', 'discarded'
    modification_details JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TYPE feedback_type AS ENUM ('rating', 'detailed_review', 'quick_thumbs', 'voice_feedback', 'behavioral');

-- Learning progression tracking
CREATE TABLE user_learning_progression (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Skill assessment
    skill_area VARCHAR(50) NOT NULL, -- 'rhythm', 'harmony', 'melody', 'arrangement'
    current_level user_skill_level,
    previous_level user_skill_level,
    
    -- Progress metrics
    accuracy_score DECIMAL(3,2),
    consistency_score DECIMAL(3,2),
    creativity_score DECIMAL(3,2),
    
    -- Learning context
    learning_session_count INTEGER,
    total_practice_time_minutes INTEGER,
    successful_generations INTEGER,
    
    -- Timestamps
    assessed_at TIMESTAMPTZ DEFAULT NOW(),
    previous_assessment_at TIMESTAMPTZ
);

-- A/B testing for optimization
CREATE TABLE ab_test_participation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Test details
    test_name VARCHAR(100) NOT NULL,
    test_variant VARCHAR(50) NOT NULL,
    participation_start TIMESTAMPTZ DEFAULT NOW(),
    participation_end TIMESTAMPTZ,
    
    -- Results
    conversion_events JSONB, -- Array of conversion events
    performance_metrics JSONB,
    user_satisfaction DECIMAL(3,2),
    
    -- Context
    user_segment VARCHAR(50),
    test_context JSONB
);
```

---

## **🛡️ SECURITY & ACCESS CONTROL**

### **🔐 Row-Level Security:**
```sql
-- Enable row-level security
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE generation_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_patterns ENABLE ROW LEVEL SECURITY;

-- Users can only access their own projects
CREATE POLICY projects_user_access ON projects
    FOR ALL
    TO authenticated_users
    USING (user_id = current_user_id() OR current_user_id() = ANY(collaborators));

-- Users can only access sessions for their projects
CREATE POLICY sessions_user_access ON generation_sessions
    FOR ALL
    TO authenticated_users
    USING (user_id = current_user_id() OR project_id IN (
        SELECT id FROM projects WHERE user_id = current_user_id() OR current_user_id() = ANY(collaborators)
    ));

-- Patterns are accessible if user has access to the session
CREATE POLICY patterns_user_access ON generated_patterns
    FOR ALL
    TO authenticated_users
    USING (session_id IN (
        SELECT id FROM generation_sessions WHERE user_id = current_user_id()
    ));

-- Function to get current user ID from JWT token
CREATE OR REPLACE FUNCTION current_user_id()
RETURNS UUID AS $$
BEGIN
    RETURN (current_setting('jwt.claims.user_id', true))::UUID;
EXCEPTION
    WHEN others THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- API key management for third-party integrations
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Key details
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL, -- Hashed API key
    key_prefix VARCHAR(20) NOT NULL, -- First few characters for identification
    
    -- Permissions
    scopes TEXT[] NOT NULL, -- ['generation', 'projects', 'collaboration']
    rate_limit_requests_per_minute INTEGER DEFAULT 60,
    
    -- Usage tracking
    total_requests INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Security
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_scopes CHECK (array_length(scopes, 1) > 0)
);

-- Index for fast API key lookup
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash) WHERE is_active = true;
CREATE INDEX idx_api_keys_user ON api_keys(user_id, is_active);
```

---

## **🚀 PERFORMANCE OPTIMIZATION**

### **⚡ Database Performance Tuning:**
```sql
-- Partitioning for large tables
CREATE TABLE generated_patterns_partitioned (
    LIKE generated_patterns INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE generated_patterns_2024_06 PARTITION OF generated_patterns_partitioned
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

-- Materialized views for common queries
CREATE MATERIALIZED VIEW user_generation_stats AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(gp.id) as total_patterns,
    AVG(gp.confidence_score) as avg_confidence,
    AVG(gp.user_rating) as avg_user_rating,
    COUNT(DISTINCT gs.id) as total_sessions,
    MAX(gs.created_at) as last_session_date,
    array_agg(DISTINCT gp.engine_name) as engines_used
FROM users u
LEFT JOIN generation_sessions gs ON u.id = gs.user_id
LEFT JOIN generated_patterns gp ON gs.id = gp.session_id
WHERE gs.created_at >= NOW() - INTERVAL '30 days'
GROUP BY u.id, u.username;

-- Refresh materialized view periodically
CREATE INDEX idx_user_stats_user_id ON user_generation_stats(user_id);

-- Connection pooling configuration
-- postgresql.conf settings:
-- max_connections = 200
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
-- work_mem = 4MB
-- maintenance_work_mem = 64MB
-- checkpoint_completion_target = 0.9

-- Query optimization with prepared statements
PREPARE frequently_used_queries;
```

### **🔄 Backup & Disaster Recovery:**
```sql
-- Point-in-time recovery setup
-- postgresql.conf:
-- wal_level = replica
-- archive_mode = on
-- archive_command = 'aws s3 cp %p s3://musearoo-backups/wal/%f'

-- Automated backup strategy
CREATE OR REPLACE FUNCTION perform_backup()
RETURNS void AS $$
BEGIN
    -- Create consistent snapshot
    PERFORM pg_start_backup('automated_backup', true);
    
    -- Backup to S3 (executed by external script)
    -- pg_dump with compression and parallel jobs
    
    PERFORM pg_stop_backup();
    
    -- Log backup completion
    INSERT INTO backup_log (backup_type, status, created_at)
    VALUES ('full', 'completed', NOW());
END;
$$ LANGUAGE plpgsql;

-- Schedule regular backups
-- Cron job: 0 2 * * * /opt/musearoo/scripts/backup.sh
```

---

## **🎼 CONCLUSION**

**MuseAroo's database architecture represents a fundamental breakthrough in musical data management.** By combining traditional relational design with cutting-edge vector embeddings, time-series optimization, and real-time state management, we've created a **data foundation that enables musical intelligence at unprecedented scale and speed**.

**Key Database Innovations:**
- ✅ **Sub-Millisecond Query Performance** - Critical musical data accessible faster than human perception
- ✅ **Musical Intelligence Storage** - AI embeddings and pattern matching for intelligent generation
- ✅ **Real-Time Collaboration** - Live session state synchronized across global users
- ✅ **Learning System Integration** - Every interaction feeds continuous AI improvement
- ✅ **Scalable Architecture** - Designed for millions of users and billions of musical patterns

**Technical Breakthroughs:**
- ✅ **PostgreSQL + TimescaleDB** - Relational and time-series data in unified system
- ✅ **Vector Similarity Search** - AI-powered pattern matching and recommendation
- ✅ **Redis Real-Time State** - Sub-10ms session state synchronization
- ✅ **Elasticsearch Musical Search** - Sophisticated musical pattern discovery
- ✅ **Row-Level Security** - Granular access control for creative intellectual property

**The data foundation that makes musical AI possible. The architectural intelligence that turns creative inspiration into organized, searchable, learnable musical knowledge. The database that grows smarter with every song.** 💾✨