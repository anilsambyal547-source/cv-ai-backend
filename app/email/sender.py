"""
Gmail API integration for sending emails to candidates.
Robust error handling, token management, and logging.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']


class GmailSender:
    """
    Handles Gmail authentication and sending emails.
    """

    def __init__(self, credentials_path: Path, token_path: Path):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None

    def authenticate(self) -> None:
        """
        Authenticate and build Gmail service.
        If token exists and is valid, reuse it; otherwise run OAuth flow.
        """
        creds = None

        # Load existing token if present
        if self.token_path.exists():
            try:
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                logger.warning(f"Failed to load token: {e}")

        # If credentials are invalid or expired, refresh or get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Refreshed expired token")
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}, will re-authenticate")
                    creds = None
            if not creds:
                # Run OAuth flow
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at {self.credentials_path}. "
                        "Download it from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
                logger.info("OAuth flow completed, new token obtained")

            # Save token for next run
            try:
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)
                logger.info(f"Token saved to {self.token_path}")
            except Exception as e:
                logger.warning(f"Failed to save token: {e}")

        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail service built successfully")

    def create_message(self, to: str, subject: str, body_text: str) -> Dict[str, Any]:
        """
        Create a raw email message ready for sending.
        """
        from email.mime.text import MIMEText
        import base64

        message = MIMEText(body_text)
        message['to'] = to
        message['from'] = 'me'  # 'me' is special value for authenticated user
        message['subject'] = subject
        # Encode in base64 (URL-safe)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {'raw': raw}

    def send_message(self, to: str, subject: str, body_text: str) -> Optional[str]:
        """
        Send an email. Returns message ID if successful, None otherwise.
        """
        if not self.service:
            self.authenticate()

        try:
            message = self.create_message(to, subject, body_text)
            sent = self.service.users().messages().send(userId='me', body=message).execute()
            logger.info(f"Email sent to {to}, message id: {sent['id']}")
            return sent['id']
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}")
            return None


# Singleton instance (lazy init)
_sender: Optional[GmailSender] = None


def get_sender(credentials_path: Path, token_path: Path) -> GmailSender:
    """Return a singleton GmailSender instance."""
    global _sender
    if _sender is None:
        _sender = GmailSender(credentials_path, token_path)
    return _sender