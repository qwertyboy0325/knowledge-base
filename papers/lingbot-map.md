---
title: "LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction"
type: paper
date: 2026-10-10
status: read
tags: [3d-reconstruction, slam, streaming, transformer, kv-cache, eccv-2026]
source: https://github.com/Robbyant/lingbot-map
summary: 前饋式串流 3D 重建基礎模型，用 anchor／局部視窗／軌跡記憶三種注意力上下文，取代 SLAM 後端優化，~20 FPS 跑上萬影格
---

# LingBot-Map

Robbyant（螞蟻集團）· ECCV 2026 · Apache-2.0
[論文 arXiv 2604.14141](https://arxiv.org/abs/2604.14141) · [Repo](https://github.com/Robbyant/lingbot-map) · [權重 HF](https://huggingface.co/robbyant/lingbot-map) · [專案頁](https://technology.robbyant.com/lingbot-map)

## TL;DR

- 輸入是影片串流，每張影格只看現在和過去。
- 輸出每張影格的相機位姿和深度圖，進而得到點雲。
- 全程只做一次 forward pass，不需要 bundle adjustment 或優化後端。
- 架構建立在 VGGT 上，關鍵是 **Geometric Context Attention (GCA)**。

## 核心想法：GCA 的三種上下文

借用經典 SLAM 的分工，但三種上下文都是端到端學出來的 attention：

| 上下文 | 對應 SLAM 概念 | 做法 |
|---|---|---|
| Anchor context | 參考座標系／尺度 | 前 n 張影格（預設 8）互相 full attention 定尺度，永不移出快取 |
| Pose-reference window | 局部地圖 | 最近 k 張影格（預設 64）保留完整 image tokens |
| Trajectory memory | 全域地圖 | 移出視窗的影格只留 6 個 token（camera、anchor、register）+ Video RoPE 時間編碼 |

舊影格的成本從約 1,005 個 token（518×378 時）降到 6 個。論文稱每影格上下文增長比 causal attention 少約 80 倍，記憶體近乎常數。

> 背景知識（Transformer、RoPE、KV cache、相機幾何、SLAM、VGGT 系譜）整理在 [前饋式 3D 重建的前置知識](../notes/3d-reconstruction-prerequisites.md)。

## 模型架構

骨架和 VGGT 幾乎一樣，差別在跨影格注意力改成串流用的因果版本，加上一套 KV cache 的保留與移出規則。參數量約 11–12 億：ViT-L ~3 億、48 個 block ~6 億、兩個 head ~2 億多。

```
影格 518×378
  │
  ▼
DINOv2 ViT-L/14（24 層）     → 37×27 = 999 個 patch token，每個 1024 維
  │ 前面接 6 個 special token：camera×1、register×4、scale(anchor)×1
  ▼  每張影格 1005 個 token
Aggregator：交替注意力 ×24 組
  ├ Frame block：只在這張影格內部做注意力，用 2D RoPE
  └ Global block：GCA，跨影格因果注意力，配 KV cache 和 3D video RoPE
  │ 取第 4、11、17、23 組的輸出，frame 和 global 串接成 2048 維
  ├──────────────────────┐
  ▼                      ▼
Camera head          DPT depth head
（只用 camera token）  （用 patch token）
  │                      │
9 維位姿               深度圖 + 信心值
平移 3、四元數 4、FoV 2
  └──────────┬───────────┘
             ▼
     深度 + 位姿反投影 → 世界座標點雲
```

**Special tokens**（`aggregator/stream.py` 的 `_setup_special_tokens`）：
- camera ×1：匯集位姿資訊，camera head 只讀它。
- register ×4：吸收注意力雜訊（DINOv2 那套機制）。
- scale ×1：即論文的 anchor token。
- 每種都學了兩組值（shape `[1, 2, N, C]`）。scale token 在前 n 張 anchor 影格用第一組、其餘用第二組；camera 和 register 只有第 1 張影格用第一組。
- 這 6 個 token 也就是軌跡記憶裡每張舊影格保留下來的部分。

**Global block（GCA）**：當前影格的 query 看 KV cache 裡的：
- anchor 影格（預設 8）的完整 token，永不移出；
- 滑動視窗（最近 64 張）的完整 token；
- 所有過去影格的 6 個 special token（只追加）。
- 3D video RoPE（`WanRotaryPosEmbed`，借自 Wan）把每個 head 的 64 維切成時間 20、高 22、寬 22。demo 預設開啟。

**Camera head（`CameraCausalHead`）**：
- 只吃最後一層的 camera token（2048 維），經 4 個 transformer block。
- 反覆精修 4 次，每次加修正量（`--camera_num_iterations`）。
- 本身也是因果的，每一輪有自己的 KV cache。

**Depth head（DPT）**：融合 4 個中間層特徵、上採樣回原解析度，輸出深度和信心值（`expp1`）。串流版預設沒開 point head，點雲用深度 + 位姿反投影。

**推論流程（`inference_streaming`）**：
1. Phase 1：前 8 張 anchor 影格一起送入，雙向 full attention 定座標和尺度。
2. Phase 2：之後一次一張；視窗滿了，最舊影格只留 6 個 special token。
   - 非 keyframe 一樣看快取、一樣輸出，只是不把自己的 KV 寫進快取（`_set_skip_append`）。
3. Windowed 模式：每個視窗清空快取，視窗間用重疊影格做 Sim(3) 對齊。

| 論文用語 | 程式碼 |
|---|---|
| Anchor context | `scale frames`、`kv_cache_scale_frames=8`、`scale_token` |
| Pose-reference window | `kv_cache_sliding_window=64` |
| Trajectory memory | `kv_cache_cross_frame_special=True`（special pages） |
| Video RoPE | `enable_3d_rope`、`WanRotaryPosEmbed` |

## 工程實作

- **骨幹：** DINOv2 ViT-L（patch 14、dim 1024），frame attention 與 global attention 交替，接 camera head 和 DPT depth head。
- **Paged KV cache：** 用 FlashInfer 實作（`lingbot_map/layers/flashinfer_cache.py`）。
  - 快取分兩個 stream：patch pages 可回收，scale 影格和滑動視窗放這裡；special-token pages 只追加、永不移出。
  - 速度約 20 FPS，連續記憶體版的 PyTorch 實作約 10.5 FPS。
  - 沒裝 FlashInfer 時可用 `--use_sdpa` 改走 SDPA。
- **訓練：**
  - 資料：31 個資料集。
  - 第一階段：離線 base model，訓練 2–24 views。
  - 第二階段：改用 GCA，影格數從 24 漸進加到 320，並加上 relative pose loss 和 context parallelism。
  - 算力：64 張 GPU，大約 3 週。
- **長序列有兩種模式：**
  - Direct mode：配合 keyframe 間隔，可撐約 3,000 張影格，約為訓練長度的 10 倍。
  - VO / windowed mode：每個視窗結束時重置狀態，視窗之間用 Sim(3) 對齊。能跑上萬張影格，但每個邊界都會累積誤差。

## 重點數據

Oxford Spires 稀疏設定（320 張影格）：

| 方法 | 類型 | AUC@15 ↑ | ATE (m) ↓ |
|---|---|---|---|
| **LingBot-Map** | online | **63.22** | **5.37** |
| VIPE | 優化式 | 45.35 | 10.52 |
| DA3 | offline | 49.84 | 12.87 |
| CUT3R | online | 5.98 | 18.16 |

- 延長到 3,840 張影格時，ATE 只從 5.37 升到 5.60；CUT3R 從 18.16 升到 32.47。
- 速度：H800 約 20 FPS，RTX 4090 約 12 FPS（518×378 解析度）。
- 消融結果：
  - Video RoPE 帶來最大的 ATE 改善（7.46 → 5.98）。
  - 視窗 64 反而比 full causal attention 更準（ATE 5.98 對 6.60），同時快 1.7 倍、省 2.7 倍記憶體。

## 限制

- 沒有明確的 loop closure。
- 不做 test-time optimization。
- VO mode 的視窗邊界誤差會隨視窗數量累積。
- 訓練時用到 320 views，KV cache 存太多影格時品質會下降，需要靠 keyframe 間隔控制。

## 我的想法 / 可以怎麼用

- 「把舊資訊壓成少數 token 再加時間編碼」的做法，和 LLM 長上下文的 KV cache 壓縮或 attention sink 的思路很像，可以對照著看。
- 上手門檻：需要 NVIDIA GPU，建議 PyTorch 2.8 + CUDA 12.8，FlashInfer 也最好裝。8 GB 顯卡可以參考 README 連結的社群 fork。
- 手機無法即時跑原模型；實際路線是手機拍、GPU 算（見討論紀錄）。
- RTX 4070 Super：建議 `--kv_cache_sliding_window 48`，估計約 6 FPS（見討論紀錄）。
- 快速試跑：`python demo.py --model_path lingbot-map.pt --image_folder example/courthouse --mask_sky`，會在 localhost:8080 開 viser 檢視器。

## 相關

- 前置知識：[前饋式 3D 重建的前置知識](../notes/3d-reconstruction-prerequisites.md)
- 基礎：VGGT、DINOv2、FlashInfer
- 同類串流方法：CUT3R、TTT3R、StreamVGGT、Stream3R、Wint3R

## 討論紀錄

<!-- 每次討論一個小節：### YYYY-MM-DD：主題 -->

### 2026-10-10：手機跑得動嗎？

**結論：原模型在手機上跑不動，移植後也做不到即時。** 最實際的是手機拍、GPU 算。數字是從架構推估，沒有實機測試。

- **軟體：** FlashInfer 只支援 CUDA；上手機要轉 Core ML／ExecuTorch，並把 paged KV cache 改成固定大小的環形緩衝區。
- **權重：** 約 11–12 億參數，FP16 ~2.3 GB、INT8 ~1.2 GB。
- **每幀運算量（518×378、k=64）：** ~10 TFLOPs。
  - 線性層：72 個 block × 1000 token × 25 MFLOPs ≈ 1.8 TFLOPs。
  - Global attention：每層 query 看 ~7.3 萬個 key，24 層合計 ≈ 7.2 TFLOPs，佔大宗。
  - 驗算：4090 跑 12 FPS ≈ 120 TFLOPS/s，和硬體規格相符。
- **KV cache：** 每張完整影格在 24 層合計 ~98 MB（1000 token × K/V × 1024 維 × 2 bytes × 24），k=64 + 8 anchor 約 7 GB。軌跡記憶每幀 ~0.6 MB。

| 設定 | 每幀運算 | KV cache | 手機估計（有效 3–8 TFLOPS/s） |
|---|---|---|---|
| 原始（518×378、k=64） | ~10 TFLOPs | ~7 GB | ~0.3–1 FPS，記憶體放不下 |
| k=16、n=2 | ~4 TFLOPs | ~1.8 GB | ~1–2 FPS，ATE 5.5 → 12.0 m |
| 再降到 ~224 解析度 | ~0.7 TFLOPs | ~0.5 GB | ~5–10 FPS，品質未知，需微調 |

**可行路線：**
1. 手機拍、伺服器算（推薦）。
2. 量化 + 降解析度 + 縮視窗硬移植，大約每秒 1–幾幀。
3. 把 ViT-L 換成 ViT-S／B 重新蒸餾，才是真正手機即時的方向，但要訓練資源。

### 2026-10-10：RTX 4070 Super 的建議設定

**結論：跑得動，大約 4090 的一半速度；12 GB 顯存跑預設設定可能 OOM，k=48 是甜蜜點。** 數字按規格比例推估，未實測。

- **算力比例：** tensor 算力約為 4090 的 43%（AI TOPS 568 對 1321），頻寬約 50%（504 對 1008 GB/s）。
- **速度：** 4090 實測 FPS（論文表 9）× 0.4–0.5：

| k | ATE（n=8） | 4090 FPS | 4070 Super 估計 | KV cache |
|---|---|---|---|---|
| 64 | 5.3 m | 12 | ~5 | ~7 GB，可能 OOM |
| **48** | **5.5 m** | 14 | **~6** | ~5.5 GB |
| 32 | 8.0 m | 15 | ~6–7 | ~4 GB |
| 16 | 11.2 m | 18 | ~8 | ~2.4 GB |

- **顯存：** 論文在 k=64 量到 13.28 GB，超過 12 GB；k=48 估計約 8–9 GB。
- **建議參數：** `--kv_cache_sliding_window 48`、`--num_scale_frames 2`（降低開頭的顯存峰值）、`--camera_num_iterations 1`（求快時用）、`--compile`。長影片用 `--mode windowed`。
- **環境：** FlashInfer 支援 Ada 架構（sm_89），但只支援 Linux。Windows 用 WSL2，或改 `--use_sdpa`。

### 2026-10-10：對 EchLub 有幫助嗎？

**結論：模型本身沒有直接幫助，不建議硬套。** 但底層知識有不少可遷移，另整理在 [串流系統的共通底層](../notes/streaming-systems-crossover.md)。

- LingBot-Map 解決的是「從影片估相機位姿和 3D 形狀」。EchLub 是協作音樂的基礎工程：Rust 權威核心、與傳輸層無關的協定、意圖式複寫、效能證據。兩者的問題領域沒有交集。
- EchLub README 明確說還沒選定產品方向，AI、即時效能等都「不是預設的下一步」。拿 3D 重建去想 AR 合奏、虛擬場館，屬於臆測，還要 GPU 伺服器，和現有的證據優先路線相衝。
- **唯一可借鏡的是一個思考框架：**「無界增長的歷史 → 固定基準 + 最近完整細節 + 舊歷史的精簡摘要」。
  - 對應到 EchLub：核心有 document ops 的 replay，操作紀錄若一路增長，重播成本也會線性增加。屆時可以考慮「快照（基準）+ 最近的完整 ops + 更早 ops 的摘要／雜湊」。
  - 但這是 event sourcing 本來就有的快照／壓縮模式，不是 LingBot-Map 發明的；真的需要時應依量測證據再做。
- **效能證據的呈現方式可參考：** 論文表 9 把「參數 → 精度／各硬體 FPS」做成網格，讓取捨一目了然，和 EchLub「先量測再承諾」的原則相容。
