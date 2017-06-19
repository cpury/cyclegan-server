import os
from flask import Flask, request, send_file
from werkzeug.exceptions import NotFound, BadRequest
from model import Model, NoFaceDetectedException, ImageUnreadableException
from io import BytesIO
from util import no_cache
import atexit

app = Flask(__name__)
app.config.from_object(__name__)

# Load default config and override config from an environment variable
app.config.update({
    'CHECKPOINT_DIR': os.path.join(app.root_path, '../model/'),
    'IMAGE_SIZE': 128,
    'MAX_CONTENT_LENGTH': 8 * 1024 * 1024,  # 8 MB
})
app.config.from_envvar('CGANSERVER_SETTINGS', silent=True)

model = Model(app.config['CHECKPOINT_DIR'], app.config['IMAGE_SIZE'])


def shutdown():
    if model:
        model.close()


atexit.register(shutdown)


@app.route('/')
def home():
    return 'ok'


@app.route('/w2m', methods=['PUT', 'POST'])
@no_cache
def w2m():
    input_file = request.files.get('file') or request.stream

    if not input_file:
        raise BadRequest(description='No file supplied')

    output_file = BytesIO()
    try:
        output = model.run_on_filedescriptor(
            'a2b', input_file, output_file, format='JPEG'
        )
    except (ImageUnreadableException, NoFaceDetectedException) as error:
        raise BadRequest(description=error.message)

    output_file.seek(0)

    return send_file(output_file, mimetype='image/jpeg')


@app.route('/m2w', methods=['PUT', 'POST'])
@no_cache
def m2w():
    input_file = request.files.get('file') or request.stream

    if not input_file:
        raise BadRequest(description='No file supplied')

    output_file = BytesIO()

    try:
        output = model.run_on_filedescriptor(
            'b2a', input_file, output_file, format='JPEG'
        )
    except (ImageUnreadableException, NoFaceDetectedException) as error:
        raise BadRequest(description=error.message)

    output_file.seek(0)

    return send_file(output_file, mimetype='image/jpeg')
