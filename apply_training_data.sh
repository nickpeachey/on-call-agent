#!/bin/bash
# Script to apply the realistic training data to PostgreSQL

echo "Copying script to PostgreSQL container..."
docker cp clear_and_regenerate.sql $(docker-compose ps -q postgres):/tmp/clear_and_regenerate.sql

echo "Executing training data generation with realistic statuses..."
docker-compose exec postgres psql -U oncall_user -d oncall_agent -f /tmp/clear_and_regenerate.sql

echo "Verifying the results..."
docker-compose exec postgres psql -U oncall_user -d oncall_agent -c "
SELECT 
    status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM incidents 
GROUP BY status 
ORDER BY count DESC;

SELECT 'Total incidents:', COUNT(*) FROM incidents;
SELECT 'Total resolutions:', COUNT(*) FROM incident_resolutions;
"
