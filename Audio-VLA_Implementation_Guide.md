# Audio-VLA: Binaural SELD + VLA Fusion Implementation Guide

> **목적**: Claude Code가 이 문서를 참고하여 Audio-VLA 시스템을 구현할 수 있도록, 아키텍처·데이터·학습·추론 파이프라인을 상세히 기술한다.

---

## 1. 프로젝트 개요

### 1.1 목표
- **Binaural SELD 모델**이 음원의 위치(azimuth, elevation)와 종류(class)를 검출
- **VLA 모델(SmolVLA)**이 이미지 + 언어 명령으로 로봇 조작 수행
- 두 모델을 **Fusion Module**로 연결하여, 언어 명령과 관련된 음원 위치를 시각 정보에 결합
- **데이터 수집 최소화**: SELD와 VLA를 각각 독립 학습 후, 얇은 Fusion Module만 소량 데이터로 학습

### 1.2 예시 시나리오
```
상황: 시야 내에 여러 핸드폰이 울리고 있음 (사이렌, 노래, 진동 등)
명령: "사이렌 소리가 울리는 핸드폰만 집어들어라"

1. SELD → 사이렌: (az=-10°, el=5°), 노래: (az=90°, el=15°), 진동: (az=45°, el=-20°)
2. Audio-Language Cross-Attention → "사이렌"과 가장 매칭되는 음원: (az=-10°, el=5°)
3. az/el → pixel 변환 → 이미지 상 (320, 240) 근처
4. Visual feature에 audio attention map 융합
5. VLA → 해당 위치의 핸드폰을 집는 action 출력
```

### 1.3 VLA 모델 선정: SmolVLA (450M)

**선정 이유:**
- **파라미터 수**: ~450M (OpenVLA 7B 대비 1/15 수준)으로 consumer GPU에서 fine-tuning 가능
- **아키텍처**: SigLIP vision encoder + SmolLM2 language decoder + Flow-Matching action expert (~100M)
- **중간 layer feature 접근 가능**: action expert가 VLM의 중간 layer N (= L/2)에서 feature를 가져가는 구조이므로, fusion module 삽입 지점이 자연스러움
- **오픈소스**: HuggingFace LeRobot 라이브러리 기반, 코드·weights·데이터 전부 공개
- **성능**: LIBERO, Meta-World 등 시뮬레이션 + 실제 환경에서 OpenVLA(7B), π₀(3.5B)와 동등 이상 성능

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Audio-VLA Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Binaural Audio ──→ [SELD Model] ──→ sound_tokens (B,N,D_s) │
│                          │                                   │
│                     peak_coords (B,N,2)  [az, el]            │
│                     class_logits (B,N,C)                     │
│                     energy (B,N,1)                            │
│                          │                                   │
│  Language ──→ [SmolVLA Language Encoder] ──→ lang_tokens      │
│                          │                  (B,T,D_l)        │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │ Audio-Language        │                        │
│              │ Cross-Attention       │──→ attn_weights (B,N)  │
│              └──────────────────────┘                        │
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │ az/el → pixel 변환    │──→ pixel_coords (B,N,2)│
│              │ (camera params)      │                        │
│              └──────────────────────┘                        │
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │ Gaussian Splatting    │──→ audio_attn_map      │
│              │ on Image Space       │    (B,H,W)             │
│              └──────────────────────┘                        │
│                          │                                   │
│  Image ──→ [SmolVLA Vision Encoder] ──→ visual_feat (B,H,W,D_v)
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │ AudioVisual Fusion   │──→ fused_feat (B,H,W,D_v)
│              │ (Gated Modulation)   │                        │
│              └──────────────────────┘                        │
│                          │                                   │
│                          ▼                                   │
│              [SmolVLA Action Expert] ──→ action (B, A)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 모듈별 상세 설계

### 3.1 SELD 모델

#### 모델 선택
- **ResNet-Conformer + EINv2** (DCASE 2024/2025 baseline 기반)
- 입력: Binaural audio (2ch) → log-mel spectrogram + ILD/IPD features
- 출력 형식: Multi-ACCDOA (다중 음원의 동시 검출 지원)

#### 출력 정의
```python
@dataclass
class SELDOutput:
    heatmap: torch.Tensor       # (B, T_frames, H_az, W_el) - optional, 시각화용
    peak_coords: torch.Tensor   # (B, N_max, 2) - azimuth, elevation in radians
    class_logits: torch.Tensor  # (B, N_max, C) - C = 음원 클래스 수 (예: 13 for STARSS23)
    energy: torch.Tensor        # (B, N_max, 1) - 각 peak의 에너지/confidence
    valid_mask: torch.Tensor    # (B, N_max) - bool, padding mask
```

#### 학습 데이터
- STARSS23 (Sony-TAu Realistic Spatial Soundscapes 2023)
- DCASE 2025 Task3 Stereo SELD Dataset
- 합성 데이터: SpatialScaper 라이브러리로 생성 가능

#### Sound Token 생성
```python
class SoundTokenEncoder(nn.Module):
    """
    SELD 출력을 (B, N, D_s) 형태의 sound_tokens으로 변환.
    각 토큰 = spatial_embed + class_embed + energy_embed
    """
    def __init__(self, D_s=256, num_classes=13):
        super().__init__()
        self.D_s = D_s
        
        # Spatial encoding: (az, el) → D_s
        # Sinusoidal positional encoding 사용 (학습 불필요)
        self.spatial_enc = SinusoidalPositionalEncoding(d_model=D_s, max_len=360)
        
        # Class embedding: soft logits → D_s
        # Hard label이 아닌 logit 분포를 projection (robust)
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, D_s),
            nn.GELU(),
            nn.Linear(D_s, D_s)
        )
        
        # Energy embedding: scalar → D_s
        self.energy_proj = nn.Linear(1, D_s)
        
        # Fusion: concat 3개 → D_s
        self.fuse = nn.Sequential(
            nn.Linear(D_s * 3, D_s),
            nn.LayerNorm(D_s),
            nn.GELU(),
            nn.Linear(D_s, D_s)
        )
    
    def forward(self, peak_coords, class_logits, energy):
        """
        Args:
            peak_coords: (B, N, 2) - azimuth, elevation in radians
            class_logits: (B, N, C) - raw logits (softmax 전)
            energy: (B, N, 1)
        Returns:
            sound_tokens: (B, N, D_s)
        """
        s = self.spatial_enc(peak_coords)       # (B, N, D_s)
        c = self.class_proj(class_logits)       # (B, N, D_s)
        e = self.energy_proj(energy)            # (B, N, D_s)
        return self.fuse(torch.cat([s, c, e], dim=-1))  # (B, N, D_s)
```

#### SinusoidalPositionalEncoding 구현
```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    (azimuth, elevation) 2D 좌표를 sinusoidal encoding으로 변환.
    NeRF의 positional encoding과 유사한 방식.
    """
    def __init__(self, d_model=256, num_frequencies=64):
        super().__init__()
        self.num_frequencies = num_frequencies
        # 2 (az, el) × num_frequencies × 2 (sin, cos) → projection to d_model
        self.proj = nn.Linear(2 * num_frequencies * 2, d_model)
        
        # Frequency bands (log-spaced)
        freqs = torch.logspace(0, np.log10(100), num_frequencies)
        self.register_buffer('freqs', freqs)
    
    def forward(self, coords):
        """
        Args:
            coords: (B, N, 2) - azimuth, elevation in radians
        Returns:
            encoding: (B, N, d_model)
        """
        B, N, _ = coords.shape
        # coords: (B, N, 2) → (B, N, 2, 1) * (num_freq,) → (B, N, 2, num_freq)
        scaled = coords.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # sin, cos → (B, N, 2, num_freq, 2) → flatten → (B, N, 2*num_freq*2)
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = encoded.reshape(B, N, -1)
        return self.proj(encoded)
```

---

### 3.2 SmolVLA 모델 (동결, 수정 없음)

#### 아키텍처 요약
```
SmolVLA (450M total)
├── Vision Encoder: SigLIP (~93M)
│   └── 512×512 image → PixelShuffle → 64 visual tokens
├── Language Decoder: SmolLM2 (~250M)
│   └── Text instruction → tokenize → decoder layers 처리
│   └── 중간 layer N (= L/2)에서 feature 추출 ← ★ fusion 삽입 지점
├── State Projector: Linear (sensorimotor state → 1 token)
└── Action Expert: Flow-Matching Transformer (~100M)
    └── Cross-attention + Self-attention interleaved
    └── 50-step action chunk 생성
```

#### Fusion을 위한 핵심 접근점
SmolVLA는 action expert가 VLM의 **중간 layer feature**를 입력으로 받음.
이 중간 feature에 audio attention을 곱하는 것이 가장 자연스러운 삽입 지점.

```python
# SmolVLA 내부 흐름 (개념적)
visual_tokens = vision_encoder(image)          # (B, 64, D_v)
lang_tokens = tokenizer(instruction)           # (B, T, D_l)
state_token = state_projector(robot_state)     # (B, 1, D_l)

# 모든 토큰 결합
combined = concat([visual_tokens, lang_tokens, state_token])  # (B, 64+T+1, D_l)

# Language decoder 중간 layer까지 처리
intermediate_feat = language_decoder[:N](combined)  # (B, 64+T+1, D_l)

# ★★★ 여기서 visual token 부분에 audio attention 적용 ★★★
visual_part = intermediate_feat[:, :64, :]  # (B, 64, D_l)
# → audio fusion 적용 후 다시 삽입

# Action expert
actions = action_expert(intermediate_feat)  # (B, 50, action_dim)
```

#### 필요 패키지
```bash
pip install lerobot
# SmolVLA 모델 로드
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

---

### 3.3 CLAP (Audio-Language Alignment)

#### 역할
- SELD의 class_logits를 텍스트와 같은 embedding space로 projection
- 이를 통해 Audio-Language Cross-Attention이 의미적으로 올바르게 동작

#### 모델 선택
- **LAION CLAP** (HTSAT-base audio encoder + RoBERTa text encoder)
- Pretrained checkpoint: `music_speech_audioset_epoch_15_esc_89.98.pt`
- ESC50 zero-shot 정확도 ~90%

#### 사용 방식
```python
import laion_clap

# CLAP 모델 로드 (동결)
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
clap_model.load_ckpt('music_speech_audioset_epoch_15_esc_89.98.pt')
clap_model.eval()
for param in clap_model.parameters():
    param.requires_grad = False

# Text embedding 추출 (language command에서)
text_embed = clap_model.get_text_embeddings(["siren sound"])  # (1, 512)

# Audio embedding: SELD class_logits → CLAP space로 projection하는 MLP 학습
class CLAPProjection(nn.Module):
    """SELD class logits를 CLAP embedding space로 매핑"""
    def __init__(self, num_classes=13, clap_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.GELU(),
            nn.Linear(256, clap_dim)
        )
    
    def forward(self, class_logits):
        # class_logits: (B, N, C) → (B, N, clap_dim)
        return self.proj(class_logits)
```

---

### 3.4 Fusion Module (학습 대상)

이 모듈이 SELD 출력과 SmolVLA를 연결하는 핵심 다리.

#### 3.4.1 Audio-Language Cross-Attention

```python
class AudioLanguageCrossAttention(nn.Module):
    """
    Sound tokens와 language tokens 사이 cross-attention으로
    language와 관련 높은 sound token을 찾는다.
    
    입력:
        sound_tokens: (B, N, D_s)   - SELD에서 생성된 음원 토큰
        lang_tokens:  (B, T, D_l)   - SmolVLA language encoder 중간 출력
    출력:
        attn_weights: (B, N)        - 각 sound token의 중요도 (softmax)
    """
    def __init__(self, D_s=256, D_l=576, D_hidden=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.D_hidden = D_hidden
        self.head_dim = D_hidden // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Sound tokens → Key
        self.proj_k = nn.Linear(D_s, D_hidden)
        # Language tokens → Query
        self.proj_q = nn.Linear(D_l, D_hidden)
        # Sound tokens → Value (for weighted audio feature)
        self.proj_v = nn.Linear(D_s, D_hidden)
        
        self.out_proj = nn.Linear(D_hidden, D_hidden)
        self.norm = nn.LayerNorm(D_hidden)
    
    def forward(self, sound_tokens, lang_tokens, sound_mask=None):
        """
        Returns:
            attn_weights: (B, N) - per-sound-token importance
            audio_context: (B, D_hidden) - weighted audio feature
        """
        B, N, _ = sound_tokens.shape
        B, T, _ = lang_tokens.shape
        
        Q = self.proj_q(lang_tokens)    # (B, T, D_hidden)
        K = self.proj_k(sound_tokens)   # (B, N, D_hidden)
        V = self.proj_v(sound_tokens)   # (B, N, D_hidden)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d_k)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_k)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_k)
        
        # Attention: (B, h, T, N)
        attn = (Q @ K.transpose(-1, -2)) * self.scale
        
        if sound_mask is not None:
            # sound_mask: (B, N) → (B, 1, 1, N)
            attn = attn.masked_fill(~sound_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = attn.softmax(dim=-1)  # (B, h, T, N)
        
        # Token-level importance: 모든 text token에 대해 평균
        attn_weights = attn.mean(dim=1).mean(dim=1)  # (B, N)
        attn_weights = attn_weights.softmax(dim=-1)
        
        # Weighted audio context
        out = (attn @ V)  # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.D_hidden)
        audio_context = self.out_proj(out).mean(dim=1)  # (B, D_hidden)
        
        return attn_weights, audio_context
```

#### 3.4.2 Spatial Projection (az/el → pixel)

```python
class AzElToPixel(nn.Module):
    """
    SELD의 azimuth/elevation 좌표를 이미지 pixel 좌표로 변환.
    카메라 intrinsic/extrinsic을 알고 있어야 함.
    
    가정:
    - 카메라와 binaural 마이크가 동일 위치 (또는 알려진 오프셋)
    - Pinhole camera model
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, peak_coords, camera_intrinsic, camera_extrinsic=None,
                img_h=512, img_w=512):
        """
        Args:
            peak_coords: (B, N, 2) - [azimuth, elevation] in radians
            camera_intrinsic: (B, 3, 3) or (3, 3) - fx, fy, cx, cy
            camera_extrinsic: (B, 4, 4) or None - world→camera transform
            img_h, img_w: 이미지 해상도
        Returns:
            pixel_coords: (B, N, 2) - [u, v] pixel coordinates
            in_frame_mask: (B, N) - bool, 이미지 프레임 내에 있는지
        """
        az = peak_coords[..., 0]  # (B, N)
        el = peak_coords[..., 1]  # (B, N)
        
        # Spherical → Cartesian (오른손 좌표계, z-forward)
        x = torch.cos(el) * torch.sin(az)
        y = -torch.sin(el)  # 이미지 y축은 아래로
        z = torch.cos(el) * torch.cos(az)
        
        points_3d = torch.stack([x, y, z], dim=-1)  # (B, N, 3)
        
        # Camera extrinsic 적용 (있을 경우)
        if camera_extrinsic is not None:
            R = camera_extrinsic[:, :3, :3]  # (B, 3, 3)
            t = camera_extrinsic[:, :3, 3:]  # (B, 3, 1)
            points_cam = (R @ points_3d.transpose(-1, -2) + t).transpose(-1, -2)
        else:
            points_cam = points_3d
        
        # Project to pixel
        # camera_intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        if camera_intrinsic.dim() == 2:
            camera_intrinsic = camera_intrinsic.unsqueeze(0).expand(points_cam.shape[0], -1, -1)
        
        proj = (camera_intrinsic @ points_cam.transpose(-1, -2)).transpose(-1, -2)  # (B, N, 3)
        
        # Perspective division
        pixel_uv = proj[..., :2] / (proj[..., 2:3] + 1e-6)  # (B, N, 2)
        
        # Frame boundary check
        in_frame = (
            (pixel_uv[..., 0] >= 0) & (pixel_uv[..., 0] < img_w) &
            (pixel_uv[..., 1] >= 0) & (pixel_uv[..., 1] < img_h) &
            (points_cam[..., 2] > 0)  # 카메라 앞에 있어야 함
        )
        
        # Clamp to image bounds
        pixel_uv = torch.stack([
            pixel_uv[..., 0].clamp(0, img_w - 1),
            pixel_uv[..., 1].clamp(0, img_h - 1)
        ], dim=-1)
        
        return pixel_uv, in_frame
```

#### 3.4.3 Audio Attention Map 생성

```python
class AudioAttentionMapGenerator(nn.Module):
    """
    Audio-language attention weights를 이미지 공간에 Gaussian으로 splatting.
    
    출력: (B, H, W) 크기의 attention map
    - 값이 높은 영역 = language 명령과 관련된 소리가 나는 위치
    """
    def __init__(self, sigma=20.0, learnable_sigma=True):
        super().__init__()
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))
        else:
            self.register_buffer('log_sigma', torch.tensor(np.log(sigma)))
    
    def forward(self, pixel_coords, attn_weights, in_frame_mask, H, W):
        """
        Args:
            pixel_coords: (B, N, 2) - [u, v]
            attn_weights: (B, N) - 각 음원의 중요도
            in_frame_mask: (B, N) - 프레임 내 여부
            H, W: attention map 해상도 (visual feature map과 동일하게)
        Returns:
            attn_map: (B, H, W) - 정규화된 attention map
        """
        B, N = attn_weights.shape
        sigma = torch.exp(self.log_sigma)
        
        # 프레임 밖 음원은 weight 0으로
        attn_weights = attn_weights * in_frame_mask.float()
        
        # Grid 생성
        device = pixel_coords.device
        grid_y = torch.arange(H, device=device).float()
        grid_x = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        
        # Gaussian splatting
        # pixel_coords: (B, N, 2) → (B, N, 1, 1, 2)
        centers = pixel_coords.unsqueeze(2).unsqueeze(3)
        # grid: (1, 1, H, W, 2)
        grid = grid.unsqueeze(0).unsqueeze(1)
        
        diff = grid - centers  # (B, N, H, W, 2)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, N, H, W)
        gaussians = torch.exp(-0.5 * sq_dist / (sigma ** 2))  # (B, N, H, W)
        
        # Weighted sum
        weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        attn_map = (weights * gaussians).sum(dim=1)  # (B, H, W)
        
        # 정규화: [0, 1]
        attn_map = attn_map / (attn_map.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        
        return attn_map
```

#### 3.4.4 AudioVisual Fusion (Gated Modulation)

```python
class AudioVisualFusion(nn.Module):
    """
    Audio attention map을 visual feature에 gate로 적용.
    
    핵심 설계:
    - audio 정보가 없는 영역 → visual feature 그대로 유지
    - audio 정보가 있는 영역 → visual feature 강조
    - 이래야 VLA의 기존 성능을 해치지 않음
    """
    def __init__(self, D_v, audio_context_dim=256):
        super().__init__()
        self.D_v = D_v
        
        # Gate network: visual feature + audio attention → gate value
        self.gate_net = nn.Sequential(
            nn.Linear(D_v + 1 + audio_context_dim, D_v),
            nn.GELU(),
            nn.Linear(D_v, D_v),
            nn.Sigmoid()
        )
        
        # Audio context를 visual feature 차원으로 projection
        self.audio_context_proj = nn.Linear(audio_context_dim, D_v)
        
        # Residual scaling (학습 초기 안정성)
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, visual_features, audio_attn_map, audio_context):
        """
        Args:
            visual_features: (B, num_tokens, D_v) - SmolVLA visual tokens
            audio_attn_map: (B, H, W) - audio attention map
            audio_context: (B, D_hidden) - weighted audio feature from cross-attn
        Returns:
            fused_features: (B, num_tokens, D_v)
        """
        B, num_tokens, D_v = visual_features.shape
        
        # SmolVLA의 visual tokens는 64개 (8×8 grid로 해석)
        H_grid = W_grid = int(num_tokens ** 0.5)  # 8
        
        # audio_attn_map을 visual token grid에 맞춤
        # (B, H, W) → (B, 1, H, W) → interpolate → (B, 1, H_grid, W_grid)
        attn_resized = F.interpolate(
            audio_attn_map.unsqueeze(1),
            size=(H_grid, W_grid),
            mode='bilinear',
            align_corners=False
        )  # (B, 1, H_grid, W_grid)
        attn_flat = attn_resized.view(B, num_tokens, 1)  # (B, 64, 1)
        
        # Audio context를 모든 토큰에 broadcast
        ac = self.audio_context_proj(audio_context)  # (B, D_v)
        ac = ac.unsqueeze(1).expand(-1, num_tokens, -1)  # (B, 64, D_v)
        
        # Gate 입력: visual feature + audio attention value + audio context
        gate_input = torch.cat([visual_features, attn_flat, ac], dim=-1)
        gate = self.gate_net(gate_input)  # (B, 64, D_v)
        
        # Gated modulation with residual scaling
        modulation = gate * attn_flat  # attention이 높은 곳에서만 활성화
        fused = visual_features + torch.sigmoid(self.alpha) * modulation * visual_features
        
        return fused
```

---

## 4. 학습 파이프라인

### 4.1 Phase 1: 각 모듈 독립 학습 (기존 데이터)

| 모듈 | 데이터 | 학습 대상 | GPU 요구량 |
|------|--------|----------|-----------|
| SELD | STARSS23, DCASE2025 Stereo SELD | 전체 모델 | 1× A100 |
| SmolVLA | 미리 학습된 checkpoint 사용 | 없음 (pretrained) | - |
| CLAP | 미리 학습된 checkpoint 사용 | 없음 (pretrained) | - |
| CLAPProjection MLP | AudioSet class labels + CLAP embeddings | MLP만 | 1× RTX 3090 |

#### CLAPProjection 학습 방법
```python
# AudioSet의 class label → CLAP text embedding을 GT로 사용
# SELD의 class logit (one-hot 기반) → CLAPProjection → CLAP text embedding과 MSE

audioset_classes = ["Speech", "Dog bark", "Siren", ...]  # 527개 or SELD 클래스 수
clap_text_embeds = clap_model.get_text_embeddings(audioset_classes)  # (C, 512)

# 학습 루프
for one_hot in class_labels:
    pred_embed = clap_projection(one_hot.unsqueeze(0))  # (1, 512)
    target_embed = clap_text_embeds[one_hot.argmax()]    # (512,)
    loss = F.mse_loss(pred_embed, target_embed)
```

### 4.2 Phase 2: Fusion Module 학습

#### 동결/학습 구분
```python
# 동결
seld_model.eval()
for p in seld_model.parameters(): p.requires_grad = False

smolvla_policy.eval()
for p in smolvla_policy.parameters(): p.requires_grad = False

clap_model.eval()
for p in clap_model.parameters(): p.requires_grad = False

# 학습 (총 파라미터: ~5-10M 수준)
trainable_modules = [
    sound_token_encoder,         # ~0.5M
    audio_language_cross_attn,   # ~1M
    audio_attn_map_generator,    # ~0 (sigma만)
    audio_visual_fusion,         # ~2M
    clap_projection,             # ~0.2M
]
optimizer = torch.optim.AdamW(
    [p for m in trainable_modules for p in m.parameters()],
    lr=1e-4, weight_decay=0.01
)
```

#### 학습 데이터 확보 전략

**1) 시뮬레이터 합성 데이터 (주력)**
```
도구: ISAAC Sim 또는 Habitat + SpatialScaper
구성:
  - Scene: 테이블 위에 물체 3-8개 배치
  - 각 물체에 AudioSet 소리 할당 (무작위)
  - Binaural rendering: RIR convolution으로 2ch 오디오 합성
  - Camera: 시뮬레이터에서 intrinsic/extrinsic 자동 제공
  - GT: (물체 위치, 음원 클래스, pixel 좌표) 자동 생성
  - Action: Scripted policy로 특정 물체 집기 → teleop 불필요

데이터 양: ~5,000-10,000 episodes (충분)
```

**2) 실제 데이터 (소량, sim-to-real gap 완화)**
```
구성:
  - Binaural 마이크 + RGB 카메라 부착 로봇
  - 핸드폰/스피커 등 소리 나는 물체 배치
  - 간단한 pick-and-place 시나리오 50-100개
  - 수동 annotation: 어떤 물체가 어떤 소리를 내는지

활용: Fusion module fine-tuning의 마지막 단계에서 사용
```

#### Loss 함수
```python
class AudioVLALoss(nn.Module):
    def __init__(self, lambda_loc=1.0, lambda_cls=0.5, lambda_action=1.0):
        super().__init__()
        self.lambda_loc = lambda_loc
        self.lambda_cls = lambda_cls
        self.lambda_action = lambda_action
    
    def forward(self, pred, target):
        """
        pred:
            attn_weights: (B, N) - 예측된 음원 중요도
            actions: (B, 50, action_dim) - 예측 action chunk
        target:
            target_sound_idx: (B,) - 정답 음원의 index
            target_actions: (B, 50, action_dim) - 정답 action
        """
        # 1. Audio grounding loss: 올바른 음원에 attention이 가도록
        grounding_loss = F.cross_entropy(
            pred['attn_weights'],  # (B, N)
            target['target_sound_idx']  # (B,) - 정답 음원 index
        )
        
        # 2. Action loss: SmolVLA의 원래 action loss (flow matching)
        action_loss = F.mse_loss(pred['actions'], target['target_actions'])
        
        total = (self.lambda_loc * grounding_loss + 
                 self.lambda_action * action_loss)
        
        return total, {
            'grounding_loss': grounding_loss.item(),
            'action_loss': action_loss.item(),
        }
```

#### 학습 하이퍼파라미터
```yaml
# config.yaml
training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  scheduler: cosine_annealing
  warmup_steps: 500
  total_steps: 50000  # ~5000 episodes × 10 epochs
  gradient_clip: 1.0

model:
  D_s: 256            # Sound token dimension
  D_l: 576            # SmolVLA language decoder hidden dim
  D_v: 576            # SmolVLA visual feature dim  
  D_hidden: 256       # Cross-attention hidden dim
  num_heads: 4        # Multi-head attention heads
  N_max: 8            # 최대 음원 수
  sigma_init: 20.0    # Gaussian splatting 초기 sigma
  num_seld_classes: 13  # STARSS23 기준

data:
  sim_episodes: 10000
  real_episodes: 100
  img_size: 512
  audio_sr: 24000
  audio_duration: 5.0  # seconds
```

---

## 5. 추론 파이프라인

```python
class AudioVLAPipeline:
    """
    전체 추론 파이프라인.
    Binaural audio + Image + Language command → Robot action
    """
    def __init__(self, config):
        # 1. 모델 로드
        self.seld = load_seld_model(config.seld_checkpoint)
        self.smolvla = SmolVLAPolicy.from_pretrained(config.smolvla_checkpoint)
        self.clap = load_clap_model(config.clap_checkpoint)
        
        # 2. Fusion modules 로드
        self.sound_token_enc = SoundTokenEncoder(D_s=config.D_s)
        self.audio_lang_cross_attn = AudioLanguageCrossAttention(
            D_s=config.D_s, D_l=config.D_l
        )
        self.azel_to_pixel = AzElToPixel()
        self.attn_map_gen = AudioAttentionMapGenerator()
        self.av_fusion = AudioVisualFusion(D_v=config.D_v)
        
        # Fusion weights 로드
        load_fusion_weights(self, config.fusion_checkpoint)
        
        # 3. 모든 모듈 eval mode
        self.eval()
    
    @torch.no_grad()
    def predict(self, binaural_audio, image, language_command, 
                camera_intrinsic, camera_extrinsic=None, robot_state=None):
        """
        Args:
            binaural_audio: (2, T_samples) - stereo audio waveform
            image: (3, H, W) - RGB image
            language_command: str - e.g., "사이렌 소리가 울리는 핸드폰을 집어라"
            camera_intrinsic: (3, 3)
            camera_extrinsic: (4, 4) or None
            robot_state: (state_dim,) or None
        Returns:
            actions: (50, action_dim) - action chunk
            debug_info: dict - 시각화용 중간 결과
        """
        # 배치 차원 추가
        audio = binaural_audio.unsqueeze(0)   # (1, 2, T)
        img = image.unsqueeze(0)              # (1, 3, H, W)
        
        # Step 1: SELD
        seld_out = self.seld(audio)
        # peak_coords: (1, N, 2), class_logits: (1, N, C), energy: (1, N, 1)
        
        # Step 2: Sound token 생성
        sound_tokens = self.sound_token_enc(
            seld_out.peak_coords, seld_out.class_logits, seld_out.energy
        )  # (1, N, D_s)
        
        # Step 3: SmolVLA language encoding (중간 layer까지)
        lang_tokens = self.smolvla.encode_language(language_command)  # (1, T, D_l)
        
        # Step 4: Audio-Language Cross-Attention
        attn_weights, audio_context = self.audio_lang_cross_attn(
            sound_tokens, lang_tokens, seld_out.valid_mask
        )  # attn_weights: (1, N), audio_context: (1, D_hidden)
        
        # Step 5: az/el → pixel 변환
        pixel_coords, in_frame = self.azel_to_pixel(
            seld_out.peak_coords, camera_intrinsic, camera_extrinsic,
            img_h=img.shape[-2], img_w=img.shape[-1]
        )  # pixel_coords: (1, N, 2)
        
        # Step 6: Audio attention map 생성
        H_feat = W_feat = 8  # SmolVLA visual token grid
        audio_attn_map = self.attn_map_gen(
            pixel_coords, attn_weights, in_frame, H_feat, W_feat
        )  # (1, H_feat, W_feat)
        
        # Step 7: SmolVLA visual encoding
        visual_features = self.smolvla.encode_vision(img)  # (1, 64, D_v)
        
        # Step 8: AudioVisual Fusion
        fused_features = self.av_fusion(
            visual_features, audio_attn_map, audio_context
        )  # (1, 64, D_v)
        
        # Step 9: SmolVLA action prediction (fused features 사용)
        actions = self.smolvla.predict_action(
            fused_features, lang_tokens, robot_state
        )  # (1, 50, action_dim)
        
        debug_info = {
            'peak_coords': seld_out.peak_coords[0],
            'class_logits': seld_out.class_logits[0],
            'attn_weights': attn_weights[0],
            'pixel_coords': pixel_coords[0],
            'audio_attn_map': audio_attn_map[0],
            'in_frame_mask': in_frame[0],
        }
        
        return actions[0], debug_info
```

---

## 6. 프로젝트 디렉토리 구조

```
audio-vla/
├── configs/
│   ├── default.yaml              # 기본 학습 설정
│   ├── sim_training.yaml         # 시뮬레이터 학습 설정
│   └── real_finetune.yaml        # 실제 데이터 fine-tune 설정
│
├── models/
│   ├── seld/
│   │   ├── resnet_conformer.py   # SELD 네트워크 (DCASE baseline 기반)
│   │   ├── seld_output.py        # SELDOutput dataclass
│   │   └── sound_token_encoder.py # Sound token 생성 모듈
│   │
│   ├── fusion/
│   │   ├── audio_language_cross_attention.py
│   │   ├── azel_to_pixel.py
│   │   ├── audio_attention_map.py
│   │   ├── audio_visual_fusion.py
│   │   └── clap_projection.py
│   │
│   ├── vla/
│   │   └── smolvla_wrapper.py    # SmolVLA 래핑 (중간 feature 추출용)
│   │
│   └── audio_vla_pipeline.py     # 전체 파이프라인 통합
│
├── data/
│   ├── sim_generator/
│   │   ├── scene_builder.py      # 시뮬레이터 scene 생성
│   │   ├── binaural_renderer.py  # RIR convolution으로 binaural 합성
│   │   └── episode_collector.py  # 에피소드 수집 스크립트
│   │
│   ├── dataset.py                # PyTorch Dataset 클래스
│   └── transforms.py             # Audio/Image augmentation
│
├── training/
│   ├── train_seld.py             # Phase 1: SELD 독립 학습
│   ├── train_clap_proj.py        # Phase 1: CLAP projection 학습
│   ├── train_fusion.py           # Phase 2: Fusion module 학습
│   └── losses.py                 # Loss 함수 정의
│
├── evaluation/
│   ├── eval_grounding.py         # Audio grounding 정확도 평가
│   ├── eval_action.py            # Action prediction 성능 평가
│   └── visualize.py              # Attention map 시각화
│
├── scripts/
│   ├── download_checkpoints.sh   # Pretrained model 다운로드
│   ├── generate_sim_data.sh      # 시뮬레이터 데이터 생성
│   └── run_inference.py          # 추론 데모
│
├── requirements.txt
└── README.md
```

---

## 7. 핵심 의존성

```txt
# requirements.txt
torch>=2.2.0
torchaudio>=2.2.0
torchvision>=0.17.0

# VLA
lerobot                   # SmolVLA 포함
transformers>=4.40.0

# SELD
soundfile
librosa>=0.10.0

# CLAP
laion-clap                # LAION CLAP

# Simulation (optional)
# isaac-sim 또는 habitat-sim

# 유틸리티
numpy
scipy
omegaconf                # config 관리
wandb                    # 학습 로깅
matplotlib               # 시각화
```

---

## 8. 구현 우선순위 및 체크리스트

### Stage 1: 기반 구축
- [ ] 프로젝트 구조 생성
- [ ] SmolVLA pretrained 모델 로드 및 중간 feature 추출 확인
- [ ] SELD baseline (DCASE2025) 코드 가져오기 + stereo 입력 동작 확인
- [ ] CLAP 모델 로드 및 text/audio embedding 추출 확인

### Stage 2: Fusion Module 구현
- [ ] `SoundTokenEncoder` 구현 + 단위 테스트
- [ ] `AudioLanguageCrossAttention` 구현 + 단위 테스트
- [ ] `AzElToPixel` 구현 + 카메라 파라미터로 검증
- [ ] `AudioAttentionMapGenerator` 구현 + 시각화 확인
- [ ] `AudioVisualFusion` 구현 + gradient flow 확인
- [ ] `CLAPProjection` 구현 + 학습

### Stage 3: 데이터 & 학습
- [ ] 시뮬레이터 데이터 생성 파이프라인 구축
- [ ] Dataset/DataLoader 구현
- [ ] Fusion module 학습 스크립트 작성
- [ ] 학습 실행 + wandb 모니터링

### Stage 4: 평가 & 디버깅
- [ ] Audio grounding 정확도 평가 (올바른 음원에 attention이 가는지)
- [ ] Action prediction 성능 평가
- [ ] Attention map 시각화로 정성적 평가
- [ ] Ablation: fusion 없는 SmolVLA vs Audio-VLA 비교

### Stage 5: Real-world (선택)
- [ ] 실제 데이터 수집 (소량)
- [ ] Sim-to-real fine-tuning
- [ ] 실제 로봇 추론 테스트

---

## 9. 알려진 리스크 및 대응

| 리스크 | 영향 | 대응 |
|-------|------|------|
| SELD az/el 정확도 부족 (10-20° 오차) | pixel 변환 후 엉뚱한 물체에 attention | Gaussian sigma를 크게 잡아 soft하게 + temporal averaging |
| 같은 방향에 물체 겹침 (depth ambiguity) | 어떤 물체가 음원인지 특정 불가 | Vision feature의 object-level 정보로 보완 |
| SmolVLA 중간 feature 접근이 어려울 경우 | Fusion 삽입 지점 변경 필요 | 대안: visual tokens에 직접 audio token을 concat |
| Sim-to-real gap | 시뮬레이터 학습 성능이 실제에서 하락 | 소량 실제 데이터 fine-tune + domain randomization |
| Audio-language alignment 부정확 | Cross-attention이 잘못된 음원 선택 | CLAP embedding space 활용 + contrastive loss 추가 |

---

## 10. 참고 자료

- **SmolVLA**: [HuggingFace Blog](https://huggingface.co/blog/smolvla), [Paper (arXiv:2506.01844)](https://arxiv.org/abs/2506.01844), [GitHub (LeRobot)](https://github.com/huggingface/lerobot)
- **SELD/DCASE**: [DCASE2025 Task3 Baseline](https://github.com/partha2409/DCASE2025_seld_baseline), [SELDnet](https://github.com/sharathadavanne/seld-dcase2022)
- **CLAP**: [LAION CLAP](https://github.com/LAION-AI/CLAP), [Microsoft CLAP](https://github.com/microsoft/CLAP)
- **OpenVLA**: [Paper (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246) - 아키텍처 참고용
