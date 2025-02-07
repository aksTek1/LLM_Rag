from datetime import datetime

class DocumentSource:
    def __init__(self, source_id: str, source_type: str, location: str):
        self.source_id = source_id
        self.source_type = source_type
        self.location = location
        self.last_updated = datetime.now()
        self.metadata = {}
