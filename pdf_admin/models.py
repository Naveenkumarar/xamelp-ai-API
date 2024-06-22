from django.db import models

# Create your models here.
from langchain.chains import ConversationalRetrievalChain

class Coversation(models.Model):
    name = models.CharField(unique=True)
    pdf = models.FileField()
    def __str__(self):
        return(str(self.name))

class Chats(models.Model):
    name = models.ForeignKey(Coversation,on_delete=models.SET_NULL, null=True)
    type = models.CharField(max_length=50,choices=(('AI','AI'),("Human","Human")))
    timestamp = models.DateTimeField(auto_now_add=True)
    message = models.CharField()
    def __str__(self) -> str:
        return (self.type+str(self.timestamp))