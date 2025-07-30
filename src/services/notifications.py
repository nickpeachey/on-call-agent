"""
Notification service for manual escalation via email and Microsoft Teams.
This module provides real implementations for sending alerts when automated 
resolution fails and manual intervention is required.
"""
import smtplib
import aiohttp
import asyncio
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for sending notifications via email and Microsoft Teams."""
    
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.teams_webhook_url = os.getenv('TEAMS_WEBHOOK_URL')
        
        # On-call contacts
        oncall_emails_str = os.getenv('ONCALL_EMAILS', '')
        self.oncall_emails = [email.strip() for email in oncall_emails_str.split(',') if email.strip()]
        
        oncall_phones_str = os.getenv('ONCALL_PHONES', '')
        self.oncall_phones = [phone.strip() for phone in oncall_phones_str.split(',') if phone.strip()]
    
    async def send_manual_escalation_alert(
        self, 
        incident: Dict[str, Any], 
        analysis: Dict[str, Any], 
        reason: str
    ) -> bool:
        """
        Send manual escalation alert via email and Teams.
        
        Args:
            incident: The incident requiring manual intervention
            analysis: AI analysis results
            reason: Reason for escalation
            
        Returns:
            bool: True if notifications sent successfully
        """
        try:
            # Prepare alert content
            alert_data = self._prepare_alert_content(incident, analysis, reason)
            
            # Send notifications concurrently
            tasks = []
            
            if self.oncall_emails and self.email_user:
                tasks.append(self._send_email_alert(alert_data))
            
            if self.teams_webhook_url:
                tasks.append(self._send_teams_alert(alert_data))
            
            if not tasks:
                logger.warning("No notification channels configured")
                return False
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if at least one notification succeeded
            success_count = sum(1 for result in results if result is True)
            
            if success_count > 0:
                logger.info(f"Manual escalation alerts sent successfully for incident {incident.get('id')}, channels: {success_count}/{len(tasks)}")
                return True
            else:
                logger.error(f"All notification channels failed for incident {incident.get('id')}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending manual escalation alert for incident {incident.get('id')}: {str(e)}")
            return False
    
    def _prepare_alert_content(
        self, 
        incident: Dict[str, Any], 
        analysis: Dict[str, Any], 
        reason: str
    ) -> Dict[str, Any]:
        """Prepare alert content for notifications."""
        
        severity_emoji = {
            'low': '🟢',
            'medium': '🟡', 
            'high': '🟠',
            'critical': '🔴'
        }
        
        confidence = analysis.get('confidence', 0.0)
        actions = analysis.get('recommended_actions', [])
        
        return {
            'incident_id': incident.get('id', 'Unknown'),
            'title': incident.get('title', 'Unknown Incident'),
            'description': incident.get('description', 'No description available'),
            'severity': incident.get('severity', 'unknown'),
            'severity_emoji': severity_emoji.get(incident.get('severity', 'unknown'), '⚪'),
            'service': incident.get('service', 'Unknown Service'),
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'confidence': f"{confidence:.2f}",
            'action_count': len(actions),
            'actions': [action.get('type', 'Unknown') for action in actions[:3]],  # Show first 3
            'logs_preview': incident.get('logs', 'No logs available')[:500]
        }
    
    async def _send_email_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send email alert to on-call team."""
        try:
            # Create email content
            subject = f"🚨 MANUAL INTERVENTION REQUIRED - {alert_data['severity_emoji']} {alert_data['title']}"
            
            html_body = f"""
            <html>
            <body>
                <h2>🚨 Manual Intervention Required</h2>
                
                <table style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Incident ID</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['incident_id']}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Title</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['title']}</td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Severity</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['severity_emoji']} {alert_data['severity'].upper()}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Service</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['service']}</td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Escalation Reason</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['reason']}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>AI Confidence</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['confidence']}</td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Available Actions</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{', '.join(alert_data['actions']) if alert_data['actions'] else 'None'}</td>
                    </tr>
                </table>
                
                <h3>Description</h3>
                <p>{alert_data['description']}</p>
                
                <h3>Log Preview</h3>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto;">
{alert_data['logs_preview']}
                </pre>
                
                <hr>
                <p><strong>Timestamp:</strong> {alert_data['timestamp']}</p>
                <p><em>This alert was sent by the AI On-Call Agent system.</em></p>
            </body>
            </html>
            """
            
            # Run email sending in thread pool to avoid blocking
            return await asyncio.get_event_loop().run_in_executor(
                None, self._send_smtp_email, subject, html_body
            )
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def _send_smtp_email(self, subject: str, html_body: str) -> bool:
        """Send email using SMTP (blocking operation)."""
        try:
            if not self.email_user or not self.email_password:
                logger.error("Email credentials not configured")
                return False
                
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.oncall_emails)
            
            # Add HTML content
            html_part = MimeText(html_body, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {len(self.oncall_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"SMTP email sending failed: {str(e)}")
            return False
    
    async def _send_teams_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send Microsoft Teams alert via webhook."""
        try:
            if not self.teams_webhook_url:
                logger.error("Teams webhook URL not configured")
                return False
                
            # Create Teams adaptive card
            card = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.2",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": "🚨 Manual Intervention Required",
                                    "weight": "Bolder",
                                    "size": "Large",
                                    "color": "Attention"
                                },
                                {
                                    "type": "FactSet",
                                    "facts": [
                                        {"title": "Incident ID", "value": alert_data['incident_id']},
                                        {"title": "Title", "value": alert_data['title']},
                                        {"title": "Severity", "value": f"{alert_data['severity_emoji']} {alert_data['severity'].upper()}"},
                                        {"title": "Service", "value": alert_data['service']},
                                        {"title": "Reason", "value": alert_data['reason']},
                                        {"title": "AI Confidence", "value": alert_data['confidence']},
                                        {"title": "Available Actions", "value": ', '.join(alert_data['actions']) if alert_data['actions'] else 'None'}
                                    ]
                                },
                                {
                                    "type": "TextBlock",
                                    "text": "**Description:**",
                                    "weight": "Bolder"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": alert_data['description'],
                                    "wrap": True
                                },
                                {
                                    "type": "TextBlock",
                                    "text": "**Log Preview:**",
                                    "weight": "Bolder"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": alert_data['logs_preview'],
                                    "fontType": "Monospace",
                                    "wrap": True
                                }
                            ]
                        }
                    }
                ]
            }
            
            # Send to Teams webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.teams_webhook_url,
                    json=card,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info("Teams alert sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Teams webhook failed with status {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Teams alert: {str(e)}")
            return False

# Global instance
notification_service = NotificationService()
