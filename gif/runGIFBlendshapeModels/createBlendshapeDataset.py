import utils
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--skipframes", type=int, default=1)
args = parser.parse_args()

path2data = "../blendshapes/"

skipframes = args.skipframes

ids, labels, groups, catgs = utils.get_vids(path2data)
utils.split_dataset(ids, labels, groups, skipframes=skipframes)
