import re
from datetime import datetime

def sanitize_string(text: str) -> str:
    return re.sub(r'[^\w\s-]', '', str(text))

def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
