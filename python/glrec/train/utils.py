import os
import math
import subprocess
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import tf_utils

import gcloud.storage as gcs
from glrec.utils import log, StopWatch


def get_distribution_strategy():
    """Detect hardware, return appropriate distribution strategy"""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        log.info('Running on  TPU ' + str(tpu.master()))
    except ValueError:
        tpu = None
        log.info('Could not connect to TPU')

    if tpu:
        try:
            log.info('Initializing TPU...')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
            log.info('TPU initialized.')
        except Exception:
            log.error('Failed to initialize TPU')
    else:
        log.info('Using default strategy for CPU and single GPU')
        distribution_strategy = tf.distribute.get_strategy()

    log.info('Replicas: ' + str(distribution_strategy.num_replicas_in_sync))
    return distribution_strategy


def download_from_gcs(bucket_name,
                      bucket_file_path,
                      local_file_path):

    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        raise RuntimeError(
            'Environment variable GOOGLE_APPLICATION_CREDENTIALS not set.')

    info_string = 'Blob gs://{}/{} downloaded to {} in:'.format(
            bucket_name, bucket_file_path, local_file_path)
    with StopWatch(info_string):
        storage_client = gcs.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(bucket_file_path)
        blob.download_to_filename(local_file_path)


def resolve_file_path(file_path):
    """
    If path is local, return the local path to file. If path is GCS, then
    download the file to /tmp/glrec/ (under the same directory structure as
    in the bucket), and return path to the local file.
    """
    gs_prefix = 'gs://'
    if file_path[:len(gs_prefix)] != gs_prefix:
        return os.path.expanduser(file_path)

    bucket_name = file_path[len(gs_prefix):].split('/')[0]
    gcs_file_path = file_path[len(gs_prefix) + len(bucket_name) + 1:]
    directory_structure = os.path.dirname(file_path[len(gs_prefix):])
    file_name = os.path.basename(file_path[len(gs_prefix):])
    local_directory = os.path.join('/tmp/glrec', directory_structure)
    if not os.path.isdir(local_directory):
        os.makedirs(local_directory)
    local_file_path = os.path.join(local_directory, file_name)
    if not os.path.isfile(local_file_path):
        download_from_gcs(bucket_name, gcs_file_path, local_file_path)
    else:
        log.info(f'Found {local_file_path} locally. No downloads needed.')
    return local_file_path


def resolve_scalars_for_tpu(parsed_yaml, num_replicas):
    """
    Finds leaf nodes in a parsed_yaml (with `yaml.safe_load`) with the
    following structure:

      key:
        v: int or float value
        tpu: scaling strategy, either 'lin', 'log', or 'sqrt'

    where `scaling_strategy` can be either 'linear' or 'logarithmic'.
    """
    if isinstance(parsed_yaml, dict):
        if set(parsed_yaml.keys()).issubset({'v', 'tpu', 't'}) and \
                {'v', 'tpu'}.issubset(set(parsed_yaml.keys())) and \
                (
                    isinstance(parsed_yaml['v'], int) or
                    isinstance(parsed_yaml['v'], float)
                ) and \
                isinstance(parsed_yaml['tpu'], str):
            if parsed_yaml['tpu'] == 'lin':
                val = type(parsed_yaml['v'])(parsed_yaml['v'] * num_replicas)
                log.debug(f'Resolved <{parsed_yaml}> to <{val}> (strat=lin)')
                return val
            elif parsed_yaml['tpu'] == 'log':
                val = type(parsed_yaml['v'])(
                        parsed_yaml['v'] * (1. + math.log(num_replicas)))
                log.debug(f'Resolved <{parsed_yaml}> to <{val}> (strat=log)')
                return val
            elif parsed_yaml['tpu'] == 'sqrt':
                val = type(parsed_yaml['v'])(
                        parsed_yaml['v'] * math.sqrt(num_replicas))
                log.debug(f'Resolved <{parsed_yaml}> to <{val}> (strat=sqrt)')
                return val
            else:
                raise ValueError('Scaling strategy (`tpu`) can only be '
                                 '"lin", "log", or "sqrt".')
        else:
            return dict([(key, resolve_scalars_for_tpu(value, num_replicas))
                        for key, value in parsed_yaml.items()])
    elif isinstance(parsed_yaml, list):
        return [resolve_scalars_for_tpu(item, num_replicas)
                for item in parsed_yaml]
    else:
        return parsed_yaml


def cmdline_sync_dir_with_gcs(local_directory, gcs_directory, wait=False):
    sync_process = subprocess.Popen([
        'gsutil', '-m', 'rsync', '-r',
        '{}'.format(local_directory),
        '{}'.format(gcs_directory)])
    if wait:
        sync_process.wait()


def resolve_training_flag(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return tf_utils.constant_value(training)
