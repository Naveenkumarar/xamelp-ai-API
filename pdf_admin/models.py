from django.db import models

# Create your models here.
from langchain.chains import ConversationalRetrievalChain

class Coversation(models.Model):
    name = models.CharField(unique=True)
    # chain = models.FileField()
    chain = models.FileField()
    def __str__(self):
        return(str(self.name))
