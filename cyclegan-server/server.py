import os
from flask import Flask, request, session, g, redirect, url_for, abort, flash
from read_cyclegan import Model
import atexit

app = Flask(__name__)
app.config.from_object(__name__)

# Load default config and override config from an environment variable
app.config.update({
    'CHECKPOINT_DIR': os.path.join(app.root_path, '../model/'),
    'IMAGE_SIZE': 256,
))
app.config.from_envvar('CGANSERVER_SETTINGS', silent=True)

model = Model(app.config.CHECKPOINT_DIR, app.config.IMAGE_SIZE)


def shutdown():
    if model:
        model.close()


atexit.register(shutdown)
