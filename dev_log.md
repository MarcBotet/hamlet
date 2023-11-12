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
* 









