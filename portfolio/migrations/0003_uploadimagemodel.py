# Generated by Django 2.2.5 on 2020-02-08 06:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0002_tensorflowmodel_load_status'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadImageModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='')),
            ],
        ),
    ]
