from tensorflow.keras import callbacks as keras_callbacks
from glrec.utils import log


def get_callback(callback, kwargs):
    if hasattr(keras_callbacks, callback):
        log.info(f'Loading `{callback}` callback from tf.keras with '
                 'parameters {kwargs}')
        callback_instance = getattr(keras_callbacks, callback)(**kwargs)
        return callback_instance
    else:
        raise ValueError('Callback `{callback}` is not supported.')
