class _TrainingLoopConstants:
    """Static object that holds information relevant for training loop"""
    def __init__(self):
        self.set_mode('gld_v2_clean')

    def set_mode(self, mode):
        supported_modes = ['gld_v1', 'gld_v2', 'gld_v2_clean']
        if mode in supported_modes:
            self._mode = mode
        else:
            raise ValueError(
                    f'`mode` should be from the set {str(supported_modes)}.')

    """
    The following values are constants used thorough the whole training
    process, regardless of the training settings.
    """

    @property
    def NUM_CLASSES(self):
        """Metadata about the Google Landmarks dataset."""
        return {
            'gld_v1': 14951,
            'gld_v2': 203094,
            'gld_v2_clean': 81313
        }[self._mode]


# Static instance to store training constants
constants = _TrainingLoopConstants()
