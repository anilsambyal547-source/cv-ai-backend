"""
Logging filter to mask Personally Identifiable Information (PII) such as email addresses and phone numbers.
"""
import re
import logging

class PIIMaskingFilter(logging.Filter):
    """
    A logging filter that redacts common PII patterns from log messages.
    """
    def __init__(self):
        super().__init__()
        # Email pattern (matches standard email addresses)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        # Phone number pattern (various international and domestic formats)
        self.phone_pattern = re.compile(
            r'(\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
        )

    def filter(self, record):
        if record.msg and isinstance(record.msg, str):
            original = record.msg
            masked = self._mask(original)
            if masked != original:
                record.msg = masked
        # If the message is not a string (e.g., exception), we skip
        return True

    def _mask(self, text):
        # Replace emails with [EMAIL]
        text = self.email_pattern.sub('[EMAIL]', text)
        # Replace phones with [PHONE]
        text = self.phone_pattern.sub('[PHONE]', text)
        return text