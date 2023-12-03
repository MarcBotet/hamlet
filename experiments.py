import itertools
import importlib
import copy


def get_model_base(architecture, backbone):
    architecture = architecture.replace("sfa_", "")
    architecture = architecture.replace("_nodbn", "")
    architecture = architecture.replace("big", "")
    architecture = architecture.replace("little", "")
    if "segformer" in architecture:
        return {
            "mitb5": f"_base_/models/{architecture}_b5.py",
            # It's intended that <=b4 refers to b5 config
            "mitb4": f"_base_/models/{architecture}_b5.py",
            "mitb3": f"_base_/models/{architecture}_b5.py",
            "mitb2": f"_base_/models/{architecture}_b5.py",
            "mitb1": f"_base_/models/{architecture}_b1.py",
            "mitb0": f"_base_/models/{architecture}_b0.py",
            "r101v1c": f"_base_/models/{architecture}_r101.py",
        }[backbone]
    if "daformer_" in architecture:
        if "mitb0" in backbone:
            return f"_base_/models/{architecture}_mitb0.py"
        if "mitb1" in backbone:
            return f"_base_/models/{architecture}_mitb1.py"
        else:
            return f"_base_/models/{architecture}_mitb5.py"

    if "upernet" in architecture and "mit" in backbone:
        return f"_base_/models/{architecture}_mit.py"
    assert "mit" not in backbone or "-del" in backbone
    return {
        "dlv2": "_base_/models/deeplabv2_r50-d8.py",
        "dlv2red": "_base_/models/deeplabv2red_r50-d8.py",
        "dlv3p": "_base_/models/deeplabv3plus_r50-d8.py",
        "da": "_base_/models/danet_r50-d8.py",
        "isa": "_base_/models/isanet_r50-d8.py",
        "uper": "_base_/models/upernet_r50.py",
    }[architecture]


def get_pretraining_file(backbone):
    if "mitb5" in backbone:
        return "pretrained/mit_b5.pth"
    if "mitb4" in backbone:
        return "pretrained/mit_b4.pth"
    if "mitb3" in backbone:
        return "pretrained/mit_b3.pth"
    if "mitb2" in backbone:
        return "pretrained/mit_b2.pth"
    if "mitb1" in backbone:
        return "pretrained/mit_b1.pth"
    if "mitb0" in backbone:
        return "pretrained/mit_b0.pth"
    if "r101v1c" in backbone:
        return "open-mmlab://resnet101_v1c"
    return {
        "r50v1c": "open-mmlab://resnet50_v1c",
        "x50-32": "open-mmlab://resnext50_32x4d",
        "x101-32": "open-mmlab://resnext101_32x4d",
        "s50": "open-mmlab://resnest50",
        "s101": "open-mmlab://resnest101",
        "s200": "open-mmlab://resnest200",
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [0, 1, 2, 3, 4, 5]:
        if backbone == f"mitb{i}":
            return dict(type=f"mit_b{i}")
        if backbone == f"mitb{i}-del":
            return dict(_delete_=True, type=f"mit_b{i}")
    return {
        "r50v1c": {"depth": 50},
        "r101v1c": {"depth": 101},
        "x50-32": {
            "type": "ResNeXt",
            "depth": 50,
            "groups": 32,
            "base_width": 4,
        },
        "x101-32": {
            "type": "ResNeXt",
            "depth": 101,
            "groups": 32,
            "base_width": 4,
        },
        "s50": {
            "type": "ResNeSt",
            "depth": 50,
            "stem_channels": 64,
            "radix": 2,
            "reduction_factor": 4,
            "avg_down_stride": True,
        },
        "s101": {
            "type": "ResNeSt",
            "depth": 101,
            "stem_channels": 128,
            "radix": 2,
            "reduction_factor": 4,
            "avg_down_stride": True,
        },
        "s200": {
            "type": "ResNeSt",
            "depth": 200,
            "stem_channels": 128,
            "radix": 2,
            "reduction_factor": 4,
            "avg_down_stride": True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault("model", {}).setdefault("decode_head", {})
    if "dlv3p" in architecture and "mit" in backbone:
        cfg["model"]["decode_head"]["c1_in_channels"] = 64
    if "sfa" in architecture:
        cfg["model"]["decode_head"]["in_channels"] = 512
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault("data", {}).setdefault("train", {})
    cfg["data"]["train"]["rare_class_sampling"] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5
    )
    return cfg


def generate_experiment_cfgs(id):
    def config_from_vars():
        tags = []

        if perfect_determinism:
            optimizer = "sgd"
            tags.append("perfect_deterministic")
        else:
            optimizer = opt

        cfg = {"_base_": ["_base_/default_runtime.py"], "n_gpus": n_gpus}
        if seed is not None:
            cfg["seed"] = seed
            cfg["deterministic"] = deterministic

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg["_base_"].append(model_base)
        cfg["model"] = {
            "pretrained": get_pretraining_file(backbone),
            "backbone": get_backbone_cfg(backbone),
        }

        loss_weight = [0, 0, 0, 1] if not train_lightweight_decoder else [1, 0, 0, 0]

        if "mit" in backbone and "original" not in architecture_mod:
            main_module = [i for i, w in enumerate(loss_weight) if w != 0][-1] + 1
            cfg["model"]["modular_training"] = modular_training
            cfg["model"]["training_policy"] = training_policy
            cfg["model"]["loss_weight"] = loss_weight
            cfg["model"]["num_module"] = main_module
            cfg["model"]["modules_update"] = modules_update
            cfg["model"]["batchnorm"] = batchnorm
            cfg["model"]["alpha"] = alpha
            cfg["model"]["mad_time_update"] = mad_time_update
            cfg["model"]["temperature"] = temperature

        if architecture_mod == "bigsegformer":
            cfg["model"]["decode_head"] = dict(
                decoder_params=dict(embed_dim=768, conv_kernel_size=1)
            )
        if architecture_mod == "littlesegformer":
            cfg["model"]["decode_head"] = dict(
                decoder_params=dict(embed_dim=256, conv_kernel_size=1)
            )
        if "sfa_" in architecture_mod:
            cfg["model"]["neck"] = dict(type="SegFormerAdapter")
        if "_nodbn" in architecture_mod:
            cfg["model"].setdefault("decode_head", {})
            cfg["model"]["decode_head"]["norm_cfg"] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # this is for warmup target or source
        cfg["segmentator_pretrained"] = pretrained_segmentator

        #!DEBUG
        cfg["freeze_backbone"] = freeze_backbone

        # Setup UDA config
        cfg["wandb_project"] = "Hamlet"
        cfg["mode"] = uda
        if uda == "target-only":
            cfg["_base_"].append(f"_base_/datasets/{target}_half_{crop}.py")
            # wandb tags
            name_ds = target
            mode = "No_UDA"
            if pretrained_segmentator is not None:
                tags.append("warmup")
        elif uda == "source-only":
            cfg["_base_"].append(f"_base_/datasets/{source}_to_{target}_{crop}.py")
            # wandb tags
            name_ds = source
            mode = "No_UDA"
            if pretrained_segmentator is not None:
                tags.append("warmup")
        elif "video" in uda:
            assert len(domain_order) == len(num_epochs)
            # eval also of the source
            path = f"configs._base_.datasets.{source}_half_{crop}"
            mod = importlib.import_module(path)
            # first position is the source dataset
            cfg_source = copy.deepcopy(mod.data["train"])
            cfg_train = dict()

            cfg_train["type"] = "SourceDataset"
            cfg_train["source"] = cfg_source
            if rcs_T is not None:
                cfg_train["rare_class_sampling"] = dict(
                    min_pixels=3000, class_temp=rcs_T, min_crop_ratio=0.5
                )
            datasets_train = [cfg_train]

            datasets_val = []
            unique_domain = set()
            for do in domain_order:
                path = f"configs._base_.datasets.{target}"
                mod = importlib.import_module(path)
                cfg_train = copy.deepcopy(mod.data["train"])
                cfg_val = copy.deepcopy(mod.data["val"])

                cfg_train["img_dir"] = cfg_train["img_dir"].replace("?", do)
                cfg_train["ann_dir"] = cfg_train["ann_dir"].replace("?", do)
                cfg_val["img_dir"] = cfg_val["img_dir"].replace("?", do)
                cfg_val["ann_dir"] = cfg_val["ann_dir"].replace("?", do)

                cfg_train["type"] = mod.dataset_type
                datasets_train.append(cfg_train)
                if do not in unique_domain:
                    datasets_val.append(cfg_val)
                unique_domain.add(do)
            cfg["online"] = dict(
                samples_per_gpu=batch_size,
                workers_per_gpu=workers_per_gpu,
                train=datasets_train,
                val=datasets_val,
                buffer=buffer,
                buffer_policy=buffer_policy,
                domain_indicator=domain_indicator,
                base_iters=base_iters,
                reduce_training=reduce_training,
                threshold_indicator=threshold_indicator,
                mode_train=mode_train,
            )
            cfg["_base_"].append(f"_base_/uda/dacs_a999_fdthings.py")
            # wandb tags
            name_ds = target
            cfg["wandb_project"] = "video-hamlet"
            cfg["domain_order"] = domain_order
            cfg["epoch_domain"] = num_epochs[0]
            mode = "online"
        elif "online" in uda:
            assert len(domain_order) == len(num_epochs)
            # eval also of the source
            path = f"configs._base_.datasets.{source}_half_{crop}"
            mod = importlib.import_module(path)
            # first position is the source dataset
            cfg_source = copy.deepcopy(mod.data["train"])
            cfg_train = dict()

            cfg_train["type"] = "SourceDataset"
            cfg_train["source"] = cfg_source
            if rcs_T is not None:
                cfg_train["rare_class_sampling"] = dict(
                    min_pixels=3000, class_temp=rcs_T, min_crop_ratio=0.5
                )
            datasets_train = [cfg_train]

            cfg_val = copy.deepcopy(mod.data["val"])
            datasets_val = [cfg_val]
            unique_domain = set()
            unique_domain.add("clear")  # which is the source
            for do in domain_order:
                path = f"configs._base_.datasets.{target}_{do}"
                mod = importlib.import_module(path)
                cfg_train = copy.deepcopy(mod.data["train"])
                cfg_val = copy.deepcopy(mod.data["val"])

                cfg_train["type"] = mod.dataset_type
                datasets_train.append(cfg_train)
                if do not in unique_domain:
                    datasets_val.append(cfg_val)
                unique_domain.add(do)
            cfg["online"] = dict(
                samples_per_gpu=batch_size,
                workers_per_gpu=workers_per_gpu,
                train=datasets_train,
                val=datasets_val,
                buffer=buffer,
                buffer_policy=buffer_policy,
                domain_indicator=domain_indicator,
                base_iters=base_iters,
                reduce_training=reduce_training,
                threshold_indicator=threshold_indicator,
                mode_train=mode_train,
            )
            cfg["_base_"].append(f"_base_/uda/dacs_a999_fdthings.py")
            # wandb tags
            name_ds = target
            cfg["wandb_project"] = "Hamlet"
            cfg["domain_order"] = domain_order
            cfg["epoch_domain"] = num_epochs[0]
            mode = "online"
        else:
            cfg["_base_"].append(f"_base_/datasets/uda_{source}_to_{target}_{crop}.py")
            cfg["_base_"].append(f"_base_/uda/{uda}.py")
            # wandb tags
            mode = "UDA"
            name_ds = "sym2real" if source == "gta" else target

        if "dacs" in uda:
            cfg.setdefault("uda", {})

        if "dacs" in uda and plcrop:
            cfg["uda"]["pseudo_weight_ignore_top"] = 15
            cfg["uda"]["pseudo_weight_ignore_bottom"] = 120
            cfg["uda"]["dynamic_dacs"] = dynamic_dacs
        cfg["data"] = dict(samples_per_gpu=batch_size, workers_per_gpu=workers_per_gpu, train={})
        if "dacs" in uda and rcs_T is not None and "online" not in uda:
            cfg = setup_rcs(cfg, rcs_T)

        # Setup optimizer and schedule
        if "dacs" in uda:
            cfg["optimizer_config"] = None  # Don't use outer optimizer

        cfg["_base_"].extend(
            [f"_base_/schedules/{optimizer}.py", f"_base_/schedules/{schedule}.py"]
        )
        cfg["optimizer"] = {"lr": lr}
        cfg["optimizer"].setdefault("paramwise_cfg", {})
        cfg["optimizer"]["paramwise_cfg"].setdefault("custom_keys", {})
        opt_param_cfg = cfg["optimizer"]["paramwise_cfg"]["custom_keys"]
        if pmult:
            opt_param_cfg["head"] = dict(lr_mult=10.0)
        if "mit" in backbone:
            opt_param_cfg["pos_block"] = dict(decay_mult=0.0)
            opt_param_cfg["norm"] = dict(decay_mult=0.0)

        # Setup runner
        if "online" in uda:
            epochs = 0
            workflow = []
            for e in num_epochs:
                epochs += e
                workflow.append(("train", e))

            cfg["runner"] = dict(type="EpochBasedRunner", max_epochs=epochs)
            cfg["workflow"] = workflow
            cfg["checkpoint_config"] = dict(by_epoch=True, interval=epochs, max_keep_ckpts=1)
            interval = 1  # if target != 'shift' else 2000
        else:
            cfg["runner"] = dict(type="IterBasedRunner", max_iters=iters)
            cfg["checkpoint_config"] = dict(by_epoch=False, interval=iters, max_keep_ckpts=1)
            interval = iters if perfect_determinism else iters // 10

        # we need efficient test to handle the 4 decoder outputs
        # efficient_test = False  # 'mit' in backbone
        efficient_test = True #!DEBUGs
        cfg["evaluation"] = dict(interval=interval, metric="mIoU", efficient_test=efficient_test)

        # Construct config name
        uda_mod = uda
        if "dacs" in uda and rcs_T is not None:
            uda_mod += f"_rcs{rcs_T}"
        if "dacs" in uda and plcrop:
            uda_mod += "_cpl"
        if "dacs" in uda:
            if "fd" in uda:
                cfg["uda"]["use_fd"] = True
            else:
                cfg["uda"]["use_fd"] = False
            uda_name = "UDA"
        else:
            uda_name = uda

        if pretrained_segmentator is not None and "dacs" in uda:
            cfg["model"]["pretrained"] = None
            cfg["uda"]["segmentator_pretrained"] = pretrained_segmentator
            cfg["uda"]["student_pretrained"] = student_pretrained
            cfg["uda"]["warmup_model"] = True
            tags.append("warmup")
        elif "dacs" in uda:
            cfg["uda"]["warmup_model"] = False

        cfg["name"] = f"{source}2{target}_{uda_mod}_{architecture_mod}_" f"{backbone}_{schedule}"

        cfg["name_mine"] = f"Hamlet_{uda_name}_{architecture_mod}_{backbone}"

        if perfect_determinism:
            cfg["name_mine"] = f"DETERMINISTIC_{uda_name}_{architecture_mod}_{backbone}"
        elif modular_training:
            tags.append("modular_training")
            tag_tr_policy = (
                f"module_{training_policy}" if isinstance(training_policy, int) else training_policy
            )
            tags.append(tag_tr_policy)
            cfg[
                "name_mine"
            ] = f"{training_policy}_{uda_name}_{architecture_mod}_{backbone}_batchnorm:{batchnorm}"
            if training_policy == "RANDOM":
                if modules_update is not None:
                    mods = modules_update.split("/")[-1].split("_")[-1].split(".npy")[0]
                    cfg["modules_update"] = mods
                    tags.append(mods)
                else:
                    raise NotImplementedError("We only support pre-generated random list")
        else:
            num_weights = [i for i, w in enumerate(loss_weight) if w != 0]
            n = num_weights[-1] + 1
            if n != 4:
                cfg["name_mine"] = f"Module_{n}_{uda_name}_{architecture_mod}_{backbone}"
                tags.append(f"module_{n}")

        if "dacs" in uda and cfg["uda"]["warmup_model"]:
            cfg["name_mine"] = f'WarmUp_{cfg["name_mine"]}'

        cfg["cfg_lr"] = dict(
            initial_lr=lr, policy_lr=policy_lr, max_lr=max_lr, lr_far_domain=lr_far_domain
        )

        cfg["exp"] = id
        cfg["name_dataset"] = f"{source}2{target}"
        cfg["name_architecture"] = f"{architecture_mod}_{backbone}"
        cfg["name_encoder"] = backbone
        cfg["name_decoder"] = architecture_mod
        cfg["name_uda"] = uda_mod
        cfg["name_opt"] = (
            f"{optimizer}_{lr}_pm{pmult}_{schedule}" f"_{n_gpus}x{batch_size}_{iters // 1000}k"
        )

        # wandb tags
        cfg["tags"] = [name_ds, mode, architecture_mod, backbone] + tags

        if "dacs" in uda:
            cfg["uda"]["use_domain_indicator"] = domain_indicator

        elif "tent" in uda:
            cfg["tent"] = True
            cfg["others"] = dict(
                type="Tent",
                segmentator_pretrained=pretrained_segmentator,
                student_pretrained=student_pretrained,
            )
            cfg["name_mine"] = "TENT"

        if "video" in uda:
            cfg["name_mine"] = f"Eval_{target}"

        custom_imports = dict(imports=["online_src.lr"], allow_failed_imports=False)
        cfg["custom_imports"] = custom_imports

        if seed is not None:
            cfg["name"] += f"_s{seed}"
        cfg["name"] = (
            cfg["name"]
            .replace(".", "")
            .replace("True", "T")
            .replace("False", "F")
            .replace("cityscapes", "cs")
            .replace("synthia", "syn")
        )
        return cfg

    # -------------------------------------------------------------------------
    # Set HAMLET defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    batch_size = 1
    opt, lr, schedule, pmult = "adamw", 0.000015, "fixed", True
    crop = "512x512"

    source, target = ("cityscapes", "rain")
    domain_order = ["clear", "25mm", "50mm", "75mm", "100mm", "200mm"] + [
        "100mm",
        "75mm",
        "50mm",
        "25mm",
        "clear",
    ]
    num_epochs = [3] * len(domain_order)
    uda = "dacs_online"

    architecture, backbone = ("segformer", "mitb1")
    pretrained_segmentator = "pretrained/mitb1_uda.pth"
    rcs_T = 0.01
    plcrop = True

    modular_training = True
    training_policy = "MAD_UP"
    train_lightweight_decoder = False
    modules_update = 4
    batchnorm = True
    alpha = 0.1
    temperature = 1.75
    mad_time_update = True

    buffer = 1000
    buffer_policy = "rare_class_sampling"

    student_pretrained = "pretrained/mitb1_modular.pth"
    dynamic_dacs = (0.5, 0.75)
    domain_indicator = True
    base_iters = 750
    reduce_training = (0.25, 0.75)
    threshold_indicator = (0.23, -0.23)

    policy_lr = "adaptive_slope"
    max_lr = 0.001
    lr_far_domain = 0.000015 * 4
    mode_train = True

    workers_per_gpu = 4
    deterministic = False
    perfect_determinism = False
    iters = 40000  # Not relevant for online training
    seed = 0

    freeze_backbone = False #!DETERMINED

    # -------------------------------------------------------------------------
    # Config experiments:
    # -------------------------------------------------------------------------
    if id == 0:
        # Hamlet parameters
        cfg = config_from_vars()
        cfgs.append(cfg)

    elif id == -1:
        import config

        seeds = config.seed
        datasets = config.datasets
        models = config.models
        udas = config.udas
        iters = config.iters
        batch_size = config.batch_size

        modular_trainings = config.modular_training
        training_policies = config.training_policy
        train_lightweight_decoder = config.train_lightweight_decoder
        batchnorm = config.batchnorm_trained

        perfect_determinism = config.perfect_determinism
        deterministic = config.deterministic

        pretrained_segmentator = config.pretrained_segmentator
        student_pretrained = config.student_pretrained

        alphas = config.alphas

        mad_time_updates = config.mad_time_update
        temperatures = config.temperature

        buffers = config.buffer
        buffer_policies = config.buffer_policy

        dynamics_dacs = config.dynamic_dacs
        domain_indicators = config.domain_indicator
        base_iterss = config.base_iters
        threshold_indicator = config.threshold_indicator

        lrs = config.lr
        lr_policies = config.lr_policy
        max_lrs = config.max_lr
        reduce_trainings = config.reduce_training
        lr_far_domains = config.lr_far_domain

        domains = config.domain_order
        epoch_num = config.num_epochs

        mode_train = config.train

        #!DEBUG
        freeze_backbone = config.freeze_backbone

        if config.modules_update is not None:
            modules_update = config.modules_update

        for (
            (source, target),
            (architecture, backbone),
            uda,
            seed,
            modular_training,
            training_policy,
            alpha,
            mad_time_update,
            temperature,
            buffer,
            buffer_policy,
            dynamic_dacs,
            domain_indicator,
            base_iters,
            lr,
            policy_lr,
            max_lr,
            reduce_training,
            lr_far_domain,
            domain_order,
        ) in itertools.product(
            datasets,
            models,
            udas,
            seeds,
            modular_trainings,
            training_policies,
            alphas,
            mad_time_updates,
            temperatures,
            buffers,
            buffer_policies,
            dynamics_dacs,
            domain_indicators,
            base_iterss,
            lrs,
            lr_policies,
            max_lrs,
            reduce_trainings,
            lr_far_domains,
            domains,
        ):
            num_epochs = [epoch_num] * len(domain_order)
            if "dacs" in uda or "online" in uda:
                rcs_T = 0.01
                plcrop = True
            else:
                rcs_T = None
                plcrop = False
            cfg = config_from_vars()
            cfgs.append(cfg)
    else:
        raise NotImplementedError("Unknown id {}".format(id))

    return cfgs
