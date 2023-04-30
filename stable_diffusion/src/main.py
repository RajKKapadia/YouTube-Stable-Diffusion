from flask import Flask, request, jsonify
from flask_cors import CORS

from stable_diffusion.model.model import generate_image
from stable_diffusion.logger import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

app = Flask(__name__)
CORS(app=app)

@app.route('/')
def home():
    return 'OK'

def get_formated_response(
    status,
    message,
    image
) -> dict:
    ''' Get formated response for the API\n
        Takes -> status, message, and base64\n
        Returns -> jsonified response
    '''
    return jsonify(
        {
            'status': status,
            'message': message,
            'image': image
        }
    )

@app.route('/api/generate', methods=['POST'])
def api_text():
    logger.info('A new request came at /api/generate')
    if request.is_json:
        body = request.get_json()
        logger.info(body)
        if 'prompt' in body.keys():
            result = generate_image(body)
            return get_formated_response(
                result['status'],
                result['message'],
                result['image']
            )
        else:
            logger.info('Request has no parameter prompt.')
            return get_formated_response(
                -1,
                'Request has no parameter prompt',
                None
            )
    else:
        logger.info('Request has no body.')
        return get_formated_response(
            -1,
            'Request has no body.',
            None
        )
