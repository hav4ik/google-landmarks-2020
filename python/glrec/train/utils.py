import tensorflow as tf
from glrec.utils import log


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


if __name__ == '__main__':
    strategy = get_distribution_strategy()
