# Generated by Django 4.1.3 on 2023-01-27 08:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("MainApp", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="chat",
            name="person",
            field=models.IntegerField(default=1),
        ),
    ]