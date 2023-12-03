
# 231103
## 베이스라인
    - 231103_1729_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_96cf4
        - domain_detector True
    - 231106_1036_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_b4e8c
    - 231106_1038_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_46eca
    - 231106_1506_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_885f2


# 231109

## 구현 및 변경사항
* `run_experiments.py`
    1) from experiments_custom import generate_experiment_cfgs as generate_experiment_cfgs_custom  
    라인 추가하여 experiments_custom으로부터 config 가져옴  
    CUSTOM boolean 변수를 통하여 True면 커스텀 코드, False면 기존 코드 사용

* `experiments_custom.py`
    1) get_model_base 함수에 mitb5_custom 항목 추가함  
    구현을 위해 `configs/_base_/models/segformer_b5_custom.py` 파일이 필요하기 때문에 만들었고 model type을 CustomEncoderDecoder로 변경함
    2) import config_custom as config 라인 추가하여 config_custom으로부터 config 가져옴

* `config_custom.py`
    1) udas = ["custom_dacs_online] 으로 구현 가능한지?
    2) modular training은 무조건 false 처리해야 할 것 같음
    3) modules_update = None 라인 필요함 (None인지 아닌지 체크하는 분기가 있음)

* CustomEncoderDecoder 클래스 구현
    1) `mmseg/models/builder.py`: elif "custom" in cfg 분기 생성  
    config의 uda = "custom_dacs_online"으로부터 이 분기까지 오도록 만들고 싶음  
    그러려면 elif "uda" in cfg보다도 elif "custom" in cfg가 먼저 오도록 해야하나? 즉 if 구문 사이에 hierarchy가 들어가게 되는건가?  
    2) `mmseg/models/segmentors/custom_encoder_decoder.py`  
    당장의 목표는 ModularEnoderDecoder과 동일한 역할을 하는 CustomEncoderDecoder 클래스를 사용해서 코드를 돌돌 돌리는 것이므로 이렇게 만듦  
    이거 연결하려면 `mmseg/models/uda/uda_decorator.py` 의 CustomUDADecorator가 필요해서 만듦  
    필요할 것 같아서 `mmseg/models/uda/dacs_custom.py` 에 CustomDACS 클래스를 만들었는데 어떻게 연결해야할지 아직 모름

## Todo
* "custom" in cfg가 True 이면서도 "uda" in cfg일 때랑 동일한 cfg를 가져가야 함  
    - 그러면 그냥 cfg["uda"]를 복붙해서 cfg["custom"]을 만들면 되는거 아닐까? ㅋㅋ
* pretrained/mitb5_uda.pth 파일을 만들어야함


# 231110
* `experiments_custom.py`
    1) get_backbone_cfg 함수 수정함: if (backbone == f"mitb{i}") | (backbone == f"mitb{i}_custom"):
    2) __cfg["custom"] = cfg["uda"].copy()__ 를 통해 일단 uda 복붙  
    이때 Config.fromfile(args.config) 라인에서 `_base_` 항목에 들어있는 파일들을 config으로 끌어들이게됨  
    근데 어떻게 "uda"에 딱 들어가지? cfg = Config.fromfile(args.config) 여기에서 뭔 마법이 일어나고있음 ;;
    3) cfg["_base_"].append(f"_base_/uda/dacs_a999_fdthings_custom.py") 라인을 통하여 custom 파일을 참조할 수 있도록 하였음

* Registry에 등록하는것은 @UDA.register_module() 와 같은 데코레이터 사용하면됨!

* ModularEncoderDecoder을 상속하는 CustomEncoderDecoder 부를때, total_modules만 선언되지 않는 이유는 무엇?
    1) 이거 조금 이상한데?

* SegFormer 공식 깃허브에서 받은 2개의 mit_b5 pretrained weights 중 64*64 사이즈 인풋을 받는 것으로 보이는 친구를 pretrained_segmentator 에 넣었더니 load state dict가 된다
    1) 얘기를 들어보니 evaluation 버전은 encoder + decoder, training 버전은 encoder only 라고 한다
    2) 근데 성능이 너무 이상함
        - ImageNet 1K pretrained면 잘하진 못하더라도 어느정도 성능은 나와야되는것 아닌가
        - 근데 전혀 못하고있음

* `run_experiments.py`
    1) CUSTOM 플래그를 args로 조정가능하도록 만들기위해 if args.custom == 0: CUSTOM = False 라인을 추가하고 add_argument를 추가했음

* CityScapes Pretraining 과정이 필요할듯...


# 231111
* SegFormer ImageNet Pretrained weights 테스트를 위해 no custom 버전에 mit b1 웨이트 끼워서 실험해보았다
    1) 시리얼
        - segformer.b1.512x512.ade.160k : 231111_1141_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_32955
        - segformer.b5.1024x1024.city.160k : 231111_1153_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_0a62b
            이딴 실수는 좀 하지말자 
        - segformer.b1.1024x1024.city.160k : 231111_1220_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_4f9b1
    2) 만약 웨이트가 문제라면 여기서 성능이 나쁘게 나올 것이고, 웨이트 문제가 아니라 코드 문제라면 여기서 성능이 괜찮을 것
    3) encoder 웨이트는 똑바로 들어가고 있는데, decoder 웨이트가 안맞음
        - /workspace/hamlet/notebooks/S00011/mitb1_error_msg.py 참조
        - Hamlet에서는 OriginalSegFormerHead 대신 SegFormerHead를 쓰고있는 걸로 파악되는데 linear fuse 모듈 등으로 인해서 디코더 웨이트가 안들어감
        - 그렇다면 mit b1 세팅에서 OSFH를 쓰면 될려나?
        - OriginalSegFormerHead : 연결안됨

* `mmseg/apis/train_sup.py`
    1) (옵션 선택) `tools/train.py`
        ```
        #!DEBUG
        # from mmseg.apis import set_random_seed, train_segmentor
        from run_experiments import CUSTOM
        if CUSTOM:
            from mmseg.apis import set_random_seed
            from mmseg.apis import train_segmentor_sup as train_segmentor
        else:
            from mmseg.apis import set_random_seed, train_segmentor
        ```
        즉 train_segmentor_sup을 쓰면 decoder training만 되도록 구현해놓음
    2) Encoder freeze
        ```
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        ```
    3) CUSTOM 모드; config_custom
        ```
        domain_order = [
            ["clear"]
            # ["clear", "25mm", "50mm", "75mm", "100mm", "200mm"] + ["100mm", "75mm", "50mm", "25mm", "clear"]
        ]
        num_epochs = 3

        models = [
            # ("segformer", "mitb5_custom"),
            ("segformer", "mitb1")
        ]
        udas = [
            "dacs_online", # Hamlet UDA
            # "custom_dacs_online"
        ]
        ...
        pretrained_segmentator = "pretrained/segformer.b1.1024x1024.city.160k.pth"

        ```
    4) 테스트
        - 최초 테스트 데모: 231111_1417_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_25e65
        - train on "clear" only
            1) 231111_1452_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_44d70 (epoch 3이라 중간에 멈춤)
            2) 231111_1521_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_58ac5 (epoch 10)
        - 근데 b5는 ModularEncoderDecoder 대신 OthersEncoderDecoder를 쓰고있기 때문에 좀더 코드를 봐야함
    5) 일단 결론
        - 아무튼 Decoder만 파인튜닝 하면 ImageNet pretrained weights는 segmentation에 쓸만하다는 결론.


# 231112
* CustomDACS
    ```
    DACS를 쓰되, backbone으로 Pretrained Foundational Encoder 끼우기?
    DACS 소스코드: https://github.com/vikolss/DACS
    지금 DACS 클래스가 ModularUDADecorator를 상속하고있기는 하지만, 필연적으로 Modular해야할 필요는 없는 거잖아?
    그럼 CustomDACS 클래스에서 Modular한 부분을 빼면 쓸수있지 않을까?
    근데 지금 세팅에서 CustomDACS를 불러올 수가 있나??
    ```

UDADecorator
get_model
extract_feat
encode_decode
forward_train
inference
simple_test
aug_test

ModularUDADecorator
freeze_or_not_modules
get_mad_info
get_main_model
get_training_policy
is_mad_training

OtherDecorator
get_main_model

DACS
get_ema_model
get_imnet_model
...

=> DACS에서 model, ema_model, imnet_model 셋다 build_segmentator를 가지고 선언하고 있음
cfg["model"] 을 잘 만들어야함 -> 기존에 만들어져있는 configs/_base_/models/segformer_b5.py 파일 활용

EncoderDecoder랑 OtherEncoderDecoder랑 대단한 차이가 없음

필요없는 arguments:
    - total_modules
    - modular_training
    - training_policy
    - loss_weight (?)
    - num_module
    - modules_update
    - batchnorm (?)
    - alpha (?)
    - mad_time_update
    - temperature (?)

CustomDACS 일단 돌돌 돌게 만들기
돌돌 돌아가긴하는데 CustomDACS의 self.student_teacher_logs()가 실행 x
SegFormerHead가 IncrementalDecodeHead를 상속하기 때문인데 이게 Modular이랑 관련이 있는지 없는지는 확인 필요
오리지널 SegFormer 코드에 없는걸 보니 hamlet 팀에서 구현한 클래스인듯
얘는 OthersEncoderDecoder랑 호환안됨 ㅠ

B5 with OthersEncoderDecoder 테스트:
- 231112_1835_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_1dad4
    - lr 0.000015
- 231112_1839_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_dc236
    - lr 0.00015
얘네 encoder freeze 안됐다 ㅋㅋ ;;

Encoder freeze 하고 다시 실험
- 231113_1219_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_251ee
    - lr 0.00015

# 231113
아놔
코딩 일부러 이따위로했나?

좀 써놓든가요;;

내가 가진 의문이 뭐냐면 이론상 모든 에폭마다 소스에 대해서는 계속 모델 업데이트가 이루어짐 (이건 질문을 해보니까 Replay Buffer라고함;;)
따라서 모든 데이터셋이 소스와 타겟을 하나씩 제공해줘야됨
근데 CityScapes라는 데이터셋의 getitem을 봐도 img, gt만 반환하지 그렇게 생긴 애가 아닌것임?
그래서 도당체 11번의 test를 하는동안 소스는 어디서 나오는지가 궁금했음

결론적으로 이론상 데이터셋을 어떻게 쓰고있냐면
일단 datasets라는 리스트에 다음 순서대로 총 12개 데이터셋들을 넣어놓음:
(소스, SourceDataset), (소스, CityscapesDataset), (25mm, CityscapesDataset), ..., (25mm, CityscapesDataset), (소스, CityscapesDataset)

이론상 이 시스템은 OnlineRunner라는 애를 만들어놓고 여기에서 training 과정을 수행하는데
이 OnlineRunner를 선언할때 이 datasets 리스트에서 0번째 element를 pop 해다가 source_dataloader이라는 이름으로 OnlineRunner에 집어넣고
그리고나서 전체 과정을 시작하는것임! 그리고 출력되는건 아마 순서대로? next를 쓰고있는걸보니 그런거같음
이걸왜 숨겨놓냐 ㅅㅂㄹ...

source free uda가 아님

# 231114

231114_0303_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_91188
- b5 originalsegformerhead 학습된거 실험
- 근데 왜 val 칼럼명이 다르지?

## 구현 및 변경사항
* `mmseg/core/evaluation/eval_hooks.py`
```
or (
    runner.model_name == "DACS"
    and runner.model.module.model_type == "OthersEncoderDecoder"
)
```
이거 변경하고 다시 실험
- 231114_1623_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_a34b1


# 231120
## 구현 및 변경사항
* `config/__base__/dataset/rain_25mm.py` 등등
    - val 데이터셋 경로 설정 별도로 해줘야함
    - baseline 실험 __231120_1314_cs2rain_dacs_online_rcs001_cpl_segformer_mitb1_fixed_s0_01ecf__

* 백본 freeze: config에서 옵셔널하게 선택할 수 있도록 구현
    - 다음 4가지 파일 모두에서 freeze_backbone 옵션 생성해야 함
        - config.py
        - config_custom.py
        - experiments.py
        - experiments_custom.py
    - `mmseg/api/train.py`
        - train_segmentor 함수에서 cfg["freeze_backbone"] 플래그 활용해서 백본 프리즈 여부 컨트롤
        ```
        if cfg["freeze_backbone"]: #!DEBUG
            for param in model.model.backbone.parameters():
                param.requires_grad = False

        ```

* OthersEncoderDecoder 활용시 EvalHook에서 제대로된 스코어 출력하도록 구현
    - 이상하게되는 실험 시리얼: 231120_1333_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_1867a
    - 보니까 ModularEncoder에서 main_model은 num_modules를 가리킴 (이거 왜 이렇게 돼있는거임? ;;)
    - 근데 OthersEncoderDecoder은 진짜로 main_model을 반환했음
    - EvalHook.evaluate() 부분에 OthersEncoderDecoder 옵셔널하게 사용하도록 별도로 if 분기를 만듦

* freeze_backbone 구현하고 다시 b5 실험
    - OthersEncoderDecoder, DACS
    - freeze_backbone = True
    - BACKBONE FROZEN: __231120_1917_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_d5f7a__
    - BACKBONE MELT: __231121_1108_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_dad32__

## Todo
* SwinTransformer 백본 합치기

# 231122
* SwinTransformer 백본 합치기

## 구현 및 변경사항
* `experiments_custom.py`
    - get_model_base
    - get_pretraining_file
    - get_backbone_cfg

아직 덜함

# 231123
* MixTransformer에 Adapter 붙이기 (1)
    - B2~B5에 붙여보자
    - Adapter from AdaptFormer? LoRA?
    - EMA가 필요한가????
    - Mix Transformer는 ViT랑 달리 매 layer마다 embedding dimension이 다르다
    - 뭔가 adapter를 가지고있는 wrapper 같은게 필요할 거 같기도?

# 231124
* MixTransformer에 Adapter 붙이기 (2)
    - LAE 논문 및 코드베이스 참고
        ```
        Adapter is a smll module that an be inserted to any layer of the pre-trained model. As shown in Fig. 2(b), the adapter is generally a residual block composed of a down-projection with parameters W_down, a nonlinear activation function sigmoid(), and an up-projection with parameters W_up. The two projections can be convolution for CNN or linear layers for Transformer architectures, respectively.
        ```

    - Adapter 붙이니까 `some parameters appear in more than one parameter group` 에러 다시 뜸
        ```
        cfg["optimizer"]["paramwise_cfg"].setdefault("custom_keys", {})
        opt_param_cfg = cfg["optimizer"]["paramwise_cfg"]["custom_keys"]
        if pmult: #!DEBUG
            opt_param_cfg["head"] = dict(lr_mult=10.0)
        if "mit" in backbone:
            opt_param_cfg["pos_block"] = dict(decay_mult=0.0)
            opt_param_cfg["norm"] = dict(decay_mult=0.0)
        ```
        - 요 라인 때문에 생기는 거 같은데 정확히 어떻게 구현되는지 몰라서 일단 주석처리하고 진행
        - cfg["optimizer"]["paramwise_cfg"] 이게 빈 딕셔너리여야 함
    
    - forward_feature 함수 변경
    - 현재 student, teacher, imnet-teacher 모두 adapter 붙는 구조인데 이거 바꿔야함


# 231201
## Recap
* SegFormer Mix Transformer 구조 관련
    - `depth`: 하나의 Stage에서 transformer block이 몇 번 반복되는지

## 구현 및 변경사항
* AdaptFormer처럼 Adapter 구현
    - Done
* Backbone Freeze 명확히 하기
    - Adapter, Decoder Head 만 requires_grad True

## 실험
* __231201_1832_cs2rain_dacs_online_rcs001_cpl_segformer_mitb5_fixed_s0_220a1__
    - Adapter, decoder head만 업데이트
    - 다른 세팅은 모두 동일

## Todo
* Decoder Head도 pretrained weights 붙일 수 있게 하기

* Static Teacher 없애버리기
    -  `configs/_base_/uda/dacs_a999_fdthings.py` 에서 imnet_feature_dist_lambda를 0으로 설정하면 imnet_head 뺄수있음
    - 근데 이 옵션이 구현은 안돼있음 개뿍침




# 231203

## 구현 및 변경사항
* 자꾸 OOM 에러가 발생해서 B5 대신 B3 세팅으로 진행
    - __231203_1147_cs2rain_dacs_online_rcs001_cpl_segformer_mitb3_fixed_s0_77b88__
        - imnet fdist 옵션 구현해서 실험
    - ??
        - imnet fdist 켜서 실험
    - 생각해보니 test time batch size를 8로 했던거 같음
    - 그리고 show_result 옵션도 꺼야겠다

* `mmseg/core/evaluation/eval_hooks.py`
    - show_result 옵션 끄기
        ```
            from mmseg.apis import single_gpu_test

            for dataloader in self.dataloaders:
                dataset_name = dataloader.dataset.name
                results = single_gpu_test(
                    runner.model,
                    dataloader,
                    # show=True,
                    show=False,
                    out_dir=self.work_dir,
                    num_epoch=runner.iter,
                    dataset_name=dataset_name,
                    img_to_pred=self.img_to_pred,
                    efficient_test=self.efficient_test,
                )
                self.evaluate(dataloader, runner, results, dataset_name)
                # ugly way to ensure ram does not crash having multiple val datasets
                gc.collect()

        ```
* `mmseg/apis/train.py`
    - test time batch size 8->1
        ```
            # register eval hooks
            if validate:
                samples = 1
                # samples = 8 #!DEBUG
                if "online" in cfg:
                    val_datasets = [build_dataset(val) for val in cfg.online.val]
                    val_dataloader = [
                        build_dataloader(
                            ds,
                            samples_per_gpu=samples,
                            workers_per_gpu=cfg.data.workers_per_gpu,
                            dist=distributed,
                            shuffle=False,
                        )
                        for ds in val_datasets
                    ]
                    eval_hook = OnlineEvalHook
                    eval_hook = eval_hook if "video" not in cfg["mode"] else VideoEvalHook
        ```

* block 1>4 -> 1>2







