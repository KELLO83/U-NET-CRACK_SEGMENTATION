# Crack Detection

# **Dataset**

crack 500 데이터셋 : https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001

참조코드 : https://github.com/yakhyo/crack-segmentation/tree/main

데이터셋 폴더 구성 : 

`CRACK500
│
├── testcrop
│
├── testdata
│
├── traincrop
│
├── traindata
│
├── valcrop
│
└── valdata`

CRACK 500 DATASET은 testcrop, testdata, traincrop, traindata, valcrop, valdata 폴더로 구성되어 있습니다. Train data, val data, testdata는 원본 이미지로, 각 이미지의 크기는 2560 x 1440입니다. Traincrop, valcrop, testcrop은 360 x 640 크기의 이미지를 포함하고 있습니다.

배치 단위로 훈련을 진행하기 위해 NVIDIA GeForce RTX 3090 환경에서 원본 이미지 데이터를 사용한 미니배치 단위 훈련을 시도하였으나, GPU 메모리 공간 부족으로 인해 CUDA ERROR가 발생했습니다. 이에 따라 360 x 640 크기의 잘려진 이미지를 사용하여 훈련을 진행하였습니다.

### Data Preprocessing

- **Padding** : crop된 이미지의 원본사이즈는 360 * 640 형태입니다
    
    ![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled.png)
    
    ![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%201.png)
    

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%202.png)

  360 x 640 원본이미지를 UNet 모델구조에 사용시 확장단계 skip connection 구조에서 tensor           shape불일치로 인한 이미지를 640 640 정사각형 형태로 만들필요가 존재합니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%203.png)

그래서 padding을 이미지와 mask에 추가해주는 코드를 작성하여 원본이미지를 640 x 640 형태로 구성해줍니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%204.png)

640 640 으로 이미지에대가 패딩을추가한후torchinfo을 통한 모델구조를 확인시 훈련가능한 param 모델구조 각 레이어 통과시 입력데이터의 형태변환을 미리 확인할수있습니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%205.png)

- **Augmentation**: 데이터의 다양성을 확보하기위해 image mask값에 대하여 수평/수직 회전을           진행합니다
    
    ![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%206.png)
    

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%207.png)

이미지는 PIL형턔의 객체이므로 tensor형태로 변형후 torch.float형태로 dtype변형을 수행을하여 model구조에 적합하게끔 데이터 전처리 과정을 수행합니다

- **Custom Dataset** :  이미지와 마스크값의 파일들을  pytorch Dataset을 구성하여 PIL객체로 이미지와 mask값을 tensor 및 데이터변형을 수행하는 CustomDataset클래스를 구성합니다
    
    ![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%208.png)
    
- **DataLoader :** 모델 훈련을 진행하기위한 DataLoaer클래스를 구성합니다 train data 는 과적합을 방지하기위해서 옵션으로 shiffle = True 옵션을 지정합니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%209.png)

### Model Training

- **Loss Function**: 손실함수로 Dice 손실함수를 사용하였습니다
- **Optimizer**: Adam optimizer를 사용하였으며 기본 Learning rate  1e-5 weight_decay = 1e-8 가중치 규제를 주어 특정가중치의 업데이트 범위를 제한하였습니다 이로써 과적합을 방지하며 전역최솟값으로 학습이되게끔 구성하였습니다
- **Learning Rate Scheduler**:학습스케쥴러 일정 횟수동안 개선이없을경우 전역최소값근처에서 발산을 방지하기위하여 loss를 기반으로 5번동안 손실률개선이 진행되지않을경우 기존 Learning rate에서 0.1만큼 곱해진 새로운 Learning rate를 구하였습니다  하나의 에폭은 여러 미니배치의 가중치업데이트로 이루어집니다 하나의 데이터셋에대하여 미니배치크기횟수만큼 가중치 업데이트가 전부수행될경우 하나의 에폭으로 가정합니다
- (`torch.optim.lr_scheduler.ReduceLROnPlateau`)
    - `mode="min"`
    - `patience=5`
    - `verbose=True`
    - `min_lr=1e-8`
    - `factor=0.1`
- **Gradient Scaler**: float16으로는 부정확하거나 범위표현이 불가한경우가 존재하여 float16으로 표시할수없는 매우작은 가중치에대하여 float32값으로 표시하여 정확도상승을 이룹니다 `torch.cuda.amp.GradScaler`
- **Metrics**: 훈련과정중 손실함수의 평가로는 검증데이터를 사용하며 평가지표으로는 Dice loss Dice score을 사용하여 개선사항을 추적합니다
    
    ![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2010.png)
    

# **U net 모델구조 설명**

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2011.png)

## **U net class**

```python
class UNet(nn.Module):
    """UNet Segmentation Model"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = DoubleConv(in_channels, out_channels=64)

        # Downscale ⬇️
        self.down1 = Down(in_channels=64, out_channels=128, scale_factor=2)  # P/2
        self.down2 = Down(in_channels=128, out_channels=256, scale_factor=2)  # P/4
        self.down3 = Down(in_channels=256, out_channels=512, scale_factor=2)  # P/8
        self.down4 = Down(in_channels=512, out_channels=1024, scale_factor=2)  # P/16

        # Upscale ⬆️
        self.up1 = Up(in_channels=1024, out_channels=512, scale_factor=2)
        self.up2 = Up(in_channels=512, out_channels=256, scale_factor=2)
        self.up3 = Up(in_channels=256, out_channels=128, scale_factor=2)
        self.up4 = Up(in_channels=128, out_channels=64, scale_factor=2)

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
  
```

 u net모델구조는 크게 4번의 수축단계와 4번의 확장단계로 구성되어있습니다 

## **Down class**

```python
class Down(nn.Module):
    """Feature Downscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor=2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)

        return 
```

다운샘플링 과정은 이미지의 특징 맵을 추출하는 단계로, nn.conv2d를 이용하여 이미지의 복잡한 특징을 추출합니다. 다운샘플링은 두 번의 conv2d 연산이 이루어진 후, maxpooling 과정을 통해 진행됩니다. 이 과정에서 scale_factor = 2를 적용하여 2x2 사이즈의 영역에서 가장 큰 값만을 하나의 값으로 구성하여 이미지의 크기를 절반으로 줄입니다. 이를 통해 이미지의 중요한 특징을 포착할 수 있습니다.

이미지 크기 변환 예시

input image size 1 * 512 * 512
nn.conv2d(in_channels=1 , out_channels=64 , kernel_size=3 , padding=1)  -> 64 * 512 * 512
nn.conv2d(in_channels=64 , out_channels = 64 , kernel_size = 3 , padding=1) -> 64 * 512 * 512
nn.MaxPool2d(kernel_size = 3 ,stride = 2) -> 64 * 256 * 256

## **Up class**

```python
class Up(nn.Module):
    """Feature Upscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=scale_factor
        )
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x_ = torch.cat([x2, x1], dim=1)
        return self.conv(x_)

```

확장 단계는 수축 단계에서 이루어진 이미지의 압축을 다시 복원시키는 과정으로, up convolution 과정을 포함합니다. 확장 단계는 up convolution, skip connection, double conv2d 과정으로 구성되며, 수축 단계와 마찬가지로 이미지의 복잡성을 포착합니다. 이 과정에서 크기가 줄여진 이미지를 원본 이미지 크기로 복원하는 역할을 수행합니다.

> Feature Map size  -> batch * 512 * 64 * 64
nn.convTranspose2d(in_channels=512 , out_channels=256 , kernel_size = 2 ,stride=2)                       -> batch * 256 * 128 * 128
concat(out,skip,dim=1)  -> batch * 512 * 128 * 128
nn.Conv2d(in_channels=512 , out_channels=256 )  -> batch * 256 * 128 * 128
nn.Conv2d(in_channles = 256 , out_channels = 256) -> batch * 256 * 128 * 128
> 

nn.ConvTranspose2d → 입력 텐서에 대한 unsample과정을 수행하는 연산입니다 

입력 이미지의 공간적 차원에 대하여 재구성에 필요한 작업입니다

- `nn.Conv2d`  CNN과정에서 이미지의 특징맵을 추출하는데 사용됩니다
- `nn.ConvTranspose2d`  이미지의 복원 및 생성 세그멘테이션 과정에서 수행합니다

nn.ConvTranspose2d 과정 입력차원 변화 확인 6*6 이미지 → 12 * 12이미지 upsample과정 수행

> Input shape 1 6 6
> 

> output shape 1 12 12
> 

> convTranspose2d weight 3 3
> 

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2012.png)

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2013.png)

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2014.png)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

2. Crack 500 Dataset을 학습 & 검출성능 분석                                                                        

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2015.png)

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2016.png)

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2017.png)

![스크린샷 2024-07-10 09-36-17.png](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-07-10_09-36-17.png)

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2018.png)

## **모델 성능평가**

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2019.png)

### Evaluation Metrics

모델 성능평가로써 Dice socre 정밀도 재현율 IOU F1 score를 사용하였습니다

- **Dice Score**: 실제예측과 ground truth 마스크값과 겹치는 정도를 측정합니다
    
    ![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2020.png)
    

- A : 모델이 예측한  영역
- B: 실제 ground truth 영역
- ∣A∩B∣예측과 실제라벨의 교차영역
- |A| and |B| 각각의 영역

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2021.png)

- **Precision**: 모델이 True라고 분류한것중에 실제 True인것의 비율입니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2022.png)

- **Recall**: 모델이 실제 True인 것중에서 모델이 True라고 예측한것의 비율입니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2023.png)

- **IoU (Intersection over Union)**: 정답영역과 예측영역이 얼만큼 겹쳐있는지 확인하는 지표입니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2024.png)

- **F1 - SCORE** : 정밀도와 재현율의 조화평균으로 구성되며 0 과 1사이의 값으로 구성합니다 1에가까울수록 분류성능이 좋습니다

![Untitled](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/Untitled%2025.png)

[[1UNET] Computer vision-based concrete crack detection using U-net fully convolutional networks.pdf](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/1UNET_Computer_vision-based_concrete_crack_detection_using_U-net_fully_convolutional_networks.pdf)

[[2DeepCrack] Learning Hierarchical Convolutional Features for Crack Detection.pdf](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/2DeepCrack_Learning_Hierarchical_Convolutional_Features_for_Crack_Detection.pdf)

[[3CrackSegNet] Image-based concrete crack detection in tunnels using deep fully convolutional networks.pdf](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/3CrackSegNet_Image-based_concrete_crack_detection_in_tunnels_using_deep_fully_convolutional_networks.pdf)

[[4DscNet] DcsNet a real-time deep network for crack segmentation.pdf](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/4DscNet_DcsNet_a_real-time_deep_network_for_crack_segmentation.pdf)

[DeepCrack_Learning_Hierarchical_Convolutional_Features_for_Crack_Detection.pdf](Crack%20Detection%2055cb69a9c8534ac39b0a639128a39ade/DeepCrack_Learning_Hierarchical_Convolutional_Features_for_Crack_Detection.pdf)