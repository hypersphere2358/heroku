# Generated by Django 2.2.5 on 2020-01-25 07:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='tensorflowmodel',
            name='load_status',
            field=models.BooleanField(default=False),
        ),
    ]
