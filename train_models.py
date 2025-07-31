#!/usr/bin/env python3
"""
Script to trigger AI model training from existing incident data
"""
import asyncio
import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import get_database_connection
from ai.simple_engine import SimpleAIEngine
from ai import AIDecisionEngine

async def train_models():
    """Train AI models from existing incident data"""
    print("🤖 Starting AI model training...")
    
    # Connect to database
    print("📡 Connecting to database...")
    db = get_database_connection()
    
    try:
        # Initialize AI engines
        print("🧠 Initializing AI engines...")
        simple_engine = SimpleAIEngine()
        ai_engine = AIDecisionEngine()
        
        # Check if we have incident data
        print("📊 Checking for training data...")
        with db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM incidents")
            incident_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM incident_resolutions")
            resolution_count = cursor.fetchone()[0]
            
        print(f"📈 Found {incident_count} incidents and {resolution_count} resolutions")
        
        if incident_count < 10:
            print("⚠️ Insufficient training data. Need at least 10 incidents.")
            return
            
        # Train Simple AI Engine
        print("🎯 Training Simple AI Engine...")
        await simple_engine.train_from_incidents()
        print("✅ Simple AI Engine training complete!")
        
        # Train Decision Engine
        print("🎯 Training AI Decision Engine...")
        await ai_engine.train_models()
        print("✅ AI Decision Engine training complete!")
        
        print("🚀 All AI models trained successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(train_models())
