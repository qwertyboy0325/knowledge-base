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

舊影格的成本從約 972 個 token 降到 6 個。論文稱每影格上下文增長比 causal attention 少約 80 倍，記憶體近乎常數。

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
- 快速試跑：`python demo.py --model_path lingbot-map.pt --image_folder example/courthouse --mask_sky`，會在 localhost:8080 開 viser 檢視器。

## 相關

- 基礎：VGGT、DINOv2、FlashInfer
- 同類串流方法：CUT3R、TTT3R、StreamVGGT、Stream3R、Wint3R
