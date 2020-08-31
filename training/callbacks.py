from tensorflow.keras import callbacks as keras_callbacks
from glrec.utils import log
from glrec.train import utils as train_utils


class GsutilRsync(keras_callbacks.Callback):
    def __init__(self, local_storage_dir, gs_storage_dir, **kwargs):
        super().__init__(**kwargs)
        self._local_storage_dir = local_storage_dir
        self._gs_storage_dir = gs_storage_dir

    def on_epoch_end(self, epoch, logs):
        train_utils.cmdline_sync_dir_with_gcs(
                self._local_storage_dir, self._gs_storage_dir)


_local_callback_mapping = {
    'GsutilRsync': GsutilRsync,
}


def get_callback(callback, kwargs):
    if hasattr(keras_callbacks, callback):
        log.info('Loading `{callback}` callback from tf.keras with '
                 'parameters {kwargs}'.format(
                     callback=callback, kwargs=kwargs))
        callback_instance = getattr(keras_callbacks, callback)(**kwargs)
        return callback_instance
    elif callback in _local_callback_mapping:
        callback_instance = _local_callback_mapping[callback](**kwargs)
    else:
        raise ValueError('Callback `{callback}` is not supported.')
