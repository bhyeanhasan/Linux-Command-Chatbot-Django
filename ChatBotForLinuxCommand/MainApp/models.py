from django.db import models


class Chat(models.Model):
    text = models.TextField(max_length=500)
    person = models.IntegerField(default=1)
