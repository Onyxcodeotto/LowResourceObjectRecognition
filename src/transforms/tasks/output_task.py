class AttachSourceMetadataTask(Task):
    
    def __init__(self, source_Type, source_identifier, name =None):
        super().__init__(name)
        self.source_type = source_Type
        self.source_identifier