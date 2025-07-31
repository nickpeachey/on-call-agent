# Copilot Instructions for AI On-Call Agent

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Context
This is an AI-powered on-call automation system for ETL infrastructure monitoring and automated issue resolution.

## Tech Stack
- **Backend**: Python with FastAPI
- **Monitoring**: Integration with logging systems (ELK, Splunk, etc.)
- **AI/ML**: OpenAI GPT, scikit-learn for pattern recognition
- **Database**: PostgreSQL for knowledge base, Redis for caching
- **Infrastructure**: Docker, monitoring Scala Spark and Airflow systems
- **APIs**: RESTful services for system integration

## Key Components
1. **Log Monitor**: Real-time log analysis and anomaly detection
2. **Knowledge Base**: Storage of known issues and their automated fixes
3. **Action Engine**: Automated execution of remediation actions (restarts, API calls)
4. **AI Decision Engine**: Pattern matching and intelligent issue classification
5. **Dashboard**: Real-time monitoring and manual override capabilities

## Coding Guidelines
- Follow async/await patterns for I/O operations
- Implement comprehensive logging and monitoring
- Use type hints for all functions and classes
- Include error handling and retry logic for all external API calls
- Write unit tests for all core functionality
- Use dependency injection for testability
- Implement circuit breaker patterns for external services
