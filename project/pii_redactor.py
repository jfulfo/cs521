import re

# TODO: more patterns?
class PIIRedactor:
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        }
    
    def apply(self, text: str) -> str:
        """Returns redacted text."""
        result = text
        result = re.sub(self.patterns['email'], '[EMAIL]', result)
        result = re.sub(self.patterns['phone'], '[PHONE]', result)
        result = re.sub(self.patterns['ssn'], '[SSN]', result)
        return result


if __name__ == "__main__":
    redactor = PIIRedactor()
    
    text = "Contact me at admin@example.com or call 555-123-4567. SSN: 123-45-6789"
    print("Original:", text)
    print("Redacted:", redactor.apply(text))