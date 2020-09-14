# Global (retrieval) heads
from .heads import ArcMarginProduct
from .heads import ArcFace
from .heads import AdaCos
from .heads import CosFace

# Feature (embedding) pooling layers
from .pooling import GeneralizedMeanPooling2D

# DELG layers
from .delg import Attention
from .delg import Autoencoder
