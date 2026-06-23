"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()

    dataset.root = ""
    dataset.kind = "imagenet"  # imagenet | latent_hier
    dataset.use_synthetic = False
    dataset.synthetic_size = 1281167
    dataset.synthetic_seed = 0

    dataset.num_workers = 16
    dataset.prefetch_factor = 8
    dataset.pin_memory = False

    dataset.image_size = 256
    dataset.image_channels = 3
    dataset.num_classes = 1000
    dataset.use_flip = False
    dataset.num_levels = 4
    dataset.class_map_path = ""
    dataset.max_open_npz = 32
    dataset.sample_single_level_per_batch = True
    dataset.level_seed = 0
    dataset.pin_levels_to_device_groups = False
    dataset.devices_per_level = 1
    dataset.level_round_robin = False

    # ------------------------------------------------------------
    # Training
    config.training = training = ml_collections.ConfigDict()

    training.learning_rate = 0.0001
    training.batch_size = 256

    training.num_epochs = 1000

    training.log_per_step = 100
    training.sample_per_epoch = 10
    training.checkpoint_per_epoch = 10
    training.fid_per_epoch = 10
    training.latent_save_per_epoch = 0
    training.latent_save_num_samples = 5
    training.latent_save_use_ema = False
    training.latent_save_ema = 0.0
    training.latent_save_png = True
    training.latent_save_npz = False
    training.latent_save_png_gap = 1
    training.half_precision = False
    
    training.optimizer_type = "muon"  # muon | adamw
    training.adam_b1 = 0.9
    training.adam_b2 = 0.95
    training.adam_wd = 0.0
    
    training.train_input_vis_num_samples_per_level = 0
    training.train_input_vis_include_prev = True
    training.train_input_vis_gap = 1

    training.seed = 42

    training.adam_b2 = 0.95
    training.ema_val = [0.9999]

    training.lr_schedule = "warmup_const"
    training.warmup_epochs = 0

    # ------------------------------------------------------------
    # MeanFlow
    config.model = model = ml_collections.ConfigDict()
    model.num_classes = dataset.num_classes
    model.input_size = dataset.image_size
    model.in_channels = dataset.image_channels
    model.num_levels = dataset.num_levels
    model.use_level_cond = False
    model.use_prev_cond = False
    model.use_token_embed = False
    model.num_level_tokens = 8
    model.cond_token_mode = "fused"  # full_concat | fused
    model.gradient_checkpointing = False

    # Architecture overrides. Use 0 / -1 to keep the model_str defaults.
    model.hidden_size = 0
    model.num_heads = 0
    model.mlp_ratio = 0.0
    model.depth = 0
    model.aux_head_depth = 8
    model.head_wide_layers = 0
    model.head_wide_size = 2048
    model.head_wide_num_heads = 16
    model.v_aux_head_depth = -1
    model.v_head_wide_layers = -1
    model.v_head_wide_size = -1
    model.v_head_wide_num_heads = -1
    model.legacy_wide_head = False

    # Noise Distribution
    model.P_mean = -0.4
    model.P_std = 1.0
    model.time_shift = False
    model.time_dist_shift = 10.0

    # Noisy conditioning (exposure bias mitigation)
    model.noisy_cond_prob = 0.0
    model.noisy_cond_alpha = 0.25

    # Loss
    model.data_proportion = 0.0 
    model.v_loss_weight = 1.0
    model.data_proportion_schedule = None  # e.g. [[0, 0.5], [30, 0.3]]
    model.cfg_beta = 1.0
    model.cfg_max = 7.0
    model.class_dropout_prob = 0.1
    model.fm_low_t_prob = 0.0
    model.use_cfg = True

    # Training Dynamics
    model.norm_p = 1.0
    model.norm_eps = 0.01
    model.struct_weight = 0.0
    model.region_var_weight = 0.0
    model.region_collapse_weight = 0.0

    # ------------------------------------------------------------
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.num_steps = 1
    sampling.num_classes = dataset.num_classes
    sampling.return_all_levels = False
    
    # ------------------------------------------------------------
    # RAE decoder 
    config.rae_decoder = rae_decoder = ml_collections.ConfigDict()
    rae_decoder.code_dir = "third_party/rae_decoder"  
    rae_decoder.enabled = False
    rae_decoder.decoder_config_path = "third_party/rae_decoder/configs/ViTXL"
    rae_decoder.pretrained_decoder_path = "checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    rae_decoder.latent_dim = 768
    rae_decoder.latent_hw = 16
    rae_decoder.decoder_patch_size = 16
    rae_decoder.image_mean = [0.485, 0.456, 0.406]
    rae_decoder.image_std = [0.229, 0.224, 0.225]
    rae_decoder.device = ""
    rae_decoder.batch_size = 16
    # Auto-fetch the decoder from HuggingFace if missing under checkpoints/rae/.
    rae_decoder.auto_download = True
    rae_decoder.hf_repo_id = "nyu-visionx/RAE-collections"


    # ------------------------------------------------------------
    # FID
    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = 50000
    fid.device_batch_size = 40
    fid.cache_ref = ""

    # ------------------------------------------------------------
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.exp_name = ""
    logging.timestamped_run_subdir = False
    logging.wandb_project = ""
    logging.wandb_entity = ""
    logging.wandb_notes = ""
    logging.wandb_tags = []

    # ------------------------------------------------------------
    # AutoGuidance
    config.autoguidance = autoguidance = ml_collections.ConfigDict()
    autoguidance.enabled = False
    autoguidance.bad_checkpoint = ""
    autoguidance.guidance_scales = [1.5]
    autoguidance.omega = 1.0
    autoguidance.t_min = 0.0
    autoguidance.t_max = 1.0

    # others
    config.load_from = ""
    config.eval_only = False
    config.load_from_folder = ""
    config.save_json = ""
    config.load_from_start_step = 6255
    config.load_from_step = 6255

    return config
