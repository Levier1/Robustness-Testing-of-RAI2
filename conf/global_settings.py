import os
from datetime import datetime


#directory to save data, results and weight files
#rotation
CHECKPOINT_PATH = '/home/featurize/work/RAI2project/RAI2checkpoint'
DATA_PATH = '/home/featurize/work/RAI2project/RAI2data'
RESULT_PATH = '/home/featurize/work/RAI2project/RAI2result'

CHECKPOINT_PATH_TRANSLATE = '/home/featurize/work/RAI2project/RAI2checkpoint'
DATA_PATH_TRANSLATE = '/home/featurize/work/RAI2project/RAI2data'
RESULT_PATH_TRANSLATE = '/home/featurize/work/RAI2project/RAI2result'

# directory to save data, results and weight files Noise and another
CHECKPOINT_PATH_NOISE = '/home/featurize/work/RAI2project/RAI2checkpoint'
DATA_PATH_NOISE = '/home/featurize/work/RAI2project/RAI2data'
RESULT_PATH_NOISE = '/home/featurize/work/RAI2project/RAI2result'

CHECKPOINT_PATH_BRIGHTNESS = '/home/featurize/work/RAI2project/RAI2checkpoint/brightness_similarity/cifar10/vgg13'
DATA_PATH_BRIGHTNESS = '/home/featurize/work/RAI2project/RAI2data'
RESULT_PATH_BRIGHTNESS = '/home/featurize/work/RAI2project/RAI2result'

CHECKPOINT_PATH_SHEAR = '/home/featurize/work/RAI2project/RAI2checkpoint/shear_similarity/cifar10/vgg13'
DATA_PATH_SHEAR = '/home/featurize/work/RAI2project/RAI2data'
RESULT_PATH_SHEAR = '/home/featurize/work/RAI2project/RAI2result'

CASE_STUDY_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, 'case_study')
CASE_STUDY_RESULT_PATH = os.path.join(RESULT_PATH, 'case_study')



#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# CIFAR100_SUBTRAIN_MEAN = [(0.5084496 , 0.48749223, 0.4421824), (0.5070316, 0.4867336, 0.4412554)]
# CIFAR100_SUBTRAIN_STD = [(0.26778224, 0.25687465, 0.2770298), (0.26830846, 0.25731015, 0.2769457)]

#mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = (0.4914008 , 0.482159  , 0.44653094)
CIFAR10_TRAIN_STD = (0.24703224, 0.24348514, 0.26158786)

# CIFAR10_SUBTRAIN_MEAN = [(0.49115923, 0.4816946 , 0.4456668), (0.49106905, 0.4816098 , 0.4461698)]
# CIFAR10_SUBTRAIN_STD = [(0.24728498, 0.24360786, 0.26152962), (0.24676642, 0.24312036, 0.26133248)]

#mean and std of TinyImagenet
TINYIMAGENET_TRAIN_MEAN = (0.48024866, 0.44807237, 0.39754647)
TINYIMAGENET_TRAIN_STD = (0.27698642, 0.2690645 , 0.2820819)

# TINYIMAGENET_SUBTRAIN_MEAN = [(0.4802226 , 0.44817278, 0.39785585), (0.4803542 , 0.4477138 , 0.39684433)]
# TINYIMAGENET_SUBTRAIN_STD = [(0.2769446 , 0.26894435, 0.28200737), (0.27689543, 0.26896143, 0.28190473)]




#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)


#total training epoches
CIFAR100_EPOCH = 200
CIFAR100_MILESTONES = [60, 120, 160]

CIFAR10_EPOCH = 100
CIFAR10_MILESTONES = [40, 70]

TINYIMAGENET_EPOCH = 200
TINYIMAGENET_MILESTONES = [60, 120, 160]

CASE_STUDY_EPOCH = 100
CASE_STUDY_MILESTONES = [60]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)


#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








