# Generated by Django 2.2.5 on 2020-01-20 12:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TensorflowModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=30)),
                ('file_path', models.CharField(max_length=30)),
                ('description', models.TextField()),
            ],
        ),
    ]
