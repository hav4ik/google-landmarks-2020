class _CompetitionConstants:
    """
    Static object that holds information about competition input
    files (such as datasets files, csv files, etc.)
    """
    def __init__(self):
        self.set_mode('kaggle')

    def set_mode(self, mode):
        supported_modes = ['kaggle', 'gcloud']
        if mode in supported_modes:
            self._mode = mode
        else:
            raise ValueError(
                    f'`mode` should be from the set {str(supported_modes)}.')

    """
    The following values with prefixes `INFER_` are used only during
    single-image inference (on Kaggle)
    """

    @property
    def INFER_DATASET_DIR(self):
        """Base dir of the data"""
        return {
            'kaggle': '../input/landmark-recognition-2020',
            'gcloud': None,
            'local': None
        }[self._mode]

    @property
    def INFER_TEST_IMAGE_DIR(self):
        """Test images base dir"""
        return {
            'kaggle': '../input/landmark-recognition-2020/test',
            'gcloud': None,
            'local': None
        }[self._mode]

    @property
    def INFER_TRAIN_IMAGE_DIR(self):
        """Train images base dir"""
        return {
            'kaggle': '../input/landmark-recognition-2020/train',
            'gcloud': None,
            'local': None
        }[self._mode]

    @property
    def TRAIN_LABELMAP_PATH(self):
        """Train set labelmap csv (id, landmark_id)"""
        return {
            'kaggle': '../input/landmark-recognition-2020/train.csv',
            'gcloud': None,
            'local': None
        }[self._mode]

    @property
    def NUM_PUBLIC_TRAIN_IMAGES(self):
        return {
            'kaggle': 1580470,
            'gcloud': 1580470,
            'local': None
        }[self._mode]


# Static instance to store the competition's constants
constants = _CompetitionConstants()
