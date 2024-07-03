DEFAULTS = {}
DEFAULTS['tridet'] = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "epic",
    "devices": ['cuda:0'],  # default: single gpu
    "train_split": ('training',),
    "val_split": ('validation',),
    "model_name": "TriDet",
    "dataset": {
        # temporal stride of the feats
        "feat_stride": 16,
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "default_fps": None,
        # input feat dim
        "input_dim": 2304,
        # number of classes
        "num_classes": 97,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        # might not be implemented by the dataloader
        "crop_ratio": None,
        # if true, force upsampling of the input features into a fixed size
        # only used for ActivityNet
        "force_upsampling": False,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (SGP | conv)
        "backbone_type": 'SGP',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_sgp_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # kernel size for boundary heads
        "boudary_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # if use the Trident-head
        "use_trident_head": True,
        # how to downsample feature in FPN
        "downsample_type": "max",
        # the K in SGP
        "k": 1.5,
        # initialization of gaussian variance for the weight in SGP
        "init_conv_vars": 0,
        # the bin number in Trident-head (exclude 0)
        "num_bins": 16,
        # the power of iou weight in loss
        "iou_weight_power": 1.,
        # add gaussian noise with the variance, play a similar role to position embedding
        "input_noise": 0,

    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0,  # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.01,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft',  # soft | hard | none
        "nms_sigma": 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh": 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW",  # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # Minimum learning rate
        "eta_min": 1e-8,
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}



DEFAULTS['actionformer'] = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "epic",
    "devices": ['cuda:0'], # default: single gpu
    "train_split": ('training', ),
    "val_split": ('validation', ),
    "model_name": "LocPointTransformer",
    "dataset": {
        # temporal stride of the feats
        "feat_stride": 16,
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "default_fps": None,
        # input feat dim
        "input_dim": 2304,
        # number of classes
        "num_classes": 97,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        # might not be implemented by the dataloader
        "crop_ratio": None,
        # if true, force upsampling of the input features into a fixed size
        # only used for ActivityNet
        "force_upsampling": False,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (convTransformer | conv)
        "backbone_type": 'convTransformer',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # number of heads in self-attention
        "n_head": 4,
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_mha_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # starting level for fpn
        "fpn_start_level": 0,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # use rel position encoding (added to self-attention)
        "use_rel_pe": False,
    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0, # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.01,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft', # soft | hard | none
        "nms_sigma" : 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh" : 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}



DEFAULTS['temporalmaxer'] = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "epic",
    "devices": ['cuda:0'],  # default: single gpu
    "train_split": ('training', ),
    "val_split": ('validation', ),
    "model_name": "TemporalMaxerArchitecture",
    "dataset": {
        # temporal stride of the feats
        "feat_stride": 16,
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "default_fps": None,
        # input feat dim
        "input_dim": 2304,
        # number of classes
        "num_classes": 97,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        # might not be implemented by the dataloader
        "crop_ratio": None,
        # if true, force upsampling of the input features into a fixed size
        # only used for ActivityNet
        "force_upsampling": False,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone
        "backbone_type": 'convPooler',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # starting level for fpn
        "fpn_start_level": 0,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
    },
    "train_cfg": {
        "loss_weight": 1.0,  # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # ratio for drop path
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.01,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft',  # soft | hard | none
        "nms_sigma": 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh": 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW",  # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}

