from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .modular_encoder_decoder import ModularEncoderDecoder
# from .others_encoder_decoder import OthersEncoderDecoder
from .custom_encoder_decoder import OthersEncoderDecoder #!DEBUG

# from run_experiments import CUSTOM
# if CUSTOM:
#     from .custom_encoder_decoder import OthersEncoderDecoder
# else:
#     from .others_encoder_decoder import OthersEncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'ModularEncoderDecoder', 'OthersEncoderDecoder']
# __all__ = ['BaseSegmentor', 'EncoderDecoder', 'ModularEncoderDecoder', 'OthersEncoderDecoder', "CustomEncoderDecoder"]

