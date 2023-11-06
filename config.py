datasets = [
    ("cityscapes", "rain"),
    # ('shift', 'shift'),
]

domain_order = [
    ["clear", "25mm", "50mm", "75mm", "100mm", "200mm"] + ["100mm", "75mm", "50mm", "25mm", "clear"]
]
num_epochs = 3

models = [
    ("segformer", "mitb1"),
]
udas = [
    "dacs_online", # Hamlet UDA
]

max_lr = [
    0.001, #1e-3
]

lr = [
    # 6e-5
    0.000015,
]

lr_policy = [
    "adaptive_slope",
]

lr_far_domain = [
    0.000015 * 4,
]

train = True

modular_training = [
    True,
]
training_policy = [  # options True:['MAD_UP', 'RANDOM', 1]
    "MAD_UP",
    # 1
]

alphas = [
    0.1,
]

batchnorm_trained = True  # Set to False to train lightweight decoder
train_lightweight_decoder = False

buffer = [
    1000,
]

buffer_policy = [
    "rare_class_sampling",
    # 'uniform'
]

temperature = [
    1.75,
]

mad_time_update = [
    True,
]

domain_indicator = [
    True,
]

dynamic_dacs = [
    # None,
    (0.5, 0.75)
]

base_iters = [
    750,
]

threshold_indicator = (0.23, -0.23)

reduce_training = [
    (0.25, 0.75),
]

batch_size = 1
iters = 40000

#modules_update = "random_modules/random_[0.25, 0.25, 0.25, 0.25].npy"
modules_update = "random_modules/online_random.npy"
pretrained_segmentator = "pretrained/mitb1_uda.pth"
# pretrained_segmentator = "pretrained/mitb1_imagenet.pth"
student_pretrained = pretrained_segmentator

seed = [0]
perfect_determinism = False
deterministic = False
