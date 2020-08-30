import os
import tensorflow as tf
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


if __name__ == '__main__':
    strategy = get_distribution_strategy()
