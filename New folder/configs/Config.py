import argparse
from collections import OrderedDict
import pprint


class TestConfig:

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default='')
        parser.add_argument('--image', type=str, default='')
        parser.add_argument('--mask', type=str, default='')
        parser.add_argument('--result', type=str, default='')
#--model = './readthis' --image './examples/places2/images' --mask './examples/places2/masks' --result './checkpoints/results'
        # Optionals
        parser.add_argument('--workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--image_size', type=int, default=(256, 256))
        parser.add_argument('--sigma', type=float, default=2.)
        parser.add_argument('--mode', type=str, default='test')
        parser.add_argument('--num_eval', type=int, default=5)

        self.opts = parser.parse_args()

    @property
    def parse(self):

        # Print Details as Ordered List
        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts
