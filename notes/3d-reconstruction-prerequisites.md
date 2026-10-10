---
title: 前饋式 3D 重建的前置知識
type: note
date: 2026-10-10
status: reading
tags: [3d-reconstruction, slam, transformer, kv-cache, prerequisites]
summary: 讀懂 LingBot-Map／VGGT 這類模型需要的背景：Transformer 與 ViT、RoPE、KV cache 與串流注意力、相機幾何、SLAM、DUSt3R→VGGT 系譜、評測指標
---

# 前饋式 3D 重建的前置知識

為了讀懂 [LingBot-Map](../papers/lingbot-map.md) 整理的背景知識，也適用於 VGGT、CUT3R、DUSt3R 這一系列的模型。

分成兩條線，最後在 VGGT 這類模型匯合：

- **A 線（深度學習）：** Transformer → ViT → DINOv2 → RoPE → KV cache 與串流注意力
- **B 線（3D 視覺）：** 相機幾何 → SfM／SLAM → DUSt3R → VGGT

每一節最後的 📍 標出這個概念在 LingBot-Map 的哪裡用到。

---

## A1. Attention 與 Transformer

- **Self-attention：** 每個 token 產生 Query、Key、Value 三個向量。
  - 用 Q·Kᵀ 算出它和其他 token 的相關程度，經 softmax 變成權重，再對 V 加權平均。
  - 直覺：每個 token 去「查詢」其他 token，把相關的資訊拉過來。
- **Multi-head：** 把維度切成好幾組（例如 1024 維切成 16 個 head，每個 64 維），各自做注意力。不同 head 可以關注不同的關係。
- **成本：** N 個 token 的注意力是 O(N²)。這是長序列、多影格時最主要的瓶頸。
- **Causal mask：** 每個 token 只能看自己和前面的 token，不能看未來。串流處理一定要用這種遮罩。
- **一個 block 的組成：** 注意力 → MLP（先放大到 4 倍維度再縮回）。每個子層都有殘差連接和 LayerNorm。
  - **QK-norm：** 先對 Q、K 做正規化再相乘，訓練比較穩定。
  - **LayerScale：** 殘差分支乘上一個從很小值（例如 0.01）開始學的係數，深層網路比較好訓練。

📍 Aggregator 的 48 個 block、camera head 的 4 個 block 都是這種結構（`lingbot_map/layers/block.py`）。

> 原始論文：Attention Is All You Need（arXiv 1706.03762）

## A2. Vision Transformer（ViT）

- **Patchify：** 把圖片切成固定大小的方塊（例如 14×14 像素），每塊攤平後經線性投影，變成一個 token。
  - 518×378 的圖、patch 14 → 37×27 = 999 個 token。
- **特殊 token：** 在 patch token 前面接幾個可學習的 token。
  - 原版 ViT 的 CLS token，用來彙整整張圖的資訊做分類。
  - VGGT／LingBot-Map 的 **camera token** 是同樣的想法，只是專門彙整「這張圖的相機位姿」。
- **Register tokens：** 論文 *Vision Transformers Need Registers*（arXiv 2309.16588）發現，ViT 會挪用一些背景 patch 當作暫存空間，導致注意力圖出現雜訊。多給幾個沒有實際意義的 register token 讓它暫存，就能解決。

📍 每張影格 = 999 個 patch token + 6 個 special token（camera 1、register 4、scale 1）。

> 原始論文：An Image is Worth 16x16 Words（arXiv 2010.11929）

## A3. DINOv2：自監督預訓練的視覺骨幹

- Meta 用大量無標註圖片做自監督訓練的 ViT，訓練方式是讓學生網路模仿老師網路、同一張圖的不同裁切要對得起來。
- 學到的特徵很「通用」，語義、對應關係、深度線索都抓得到。所以幾乎所有現代 3D 前饋模型都用它當骨幹。
- 版本：ViT-S／B／L／g，大約 2,100 萬到 11 億參數。`_reg` 結尾的是帶 register 的版本。

📍 LingBot-Map 用 `dinov2_vitl14_reg` 當編碼器，aggregator 的 block 也從它的權重初始化。

> 論文：DINOv2（arXiv 2304.07193）

## A4. 位置編碼與 RoPE

- **為什麼需要位置編碼：** 注意力本身不知道 token 的順序或位置，必須另外告訴它。
- **RoPE（Rotary Position Embedding）：**
  - 把 Q、K 的維度兩兩一組，看成平面上的向量，依位置旋轉一個角度。
  - 兩個 token 做內積時，結果只和**相對位置**有關。
  - 好處：自然表達「距離多遠」，序列長度也比較能外推。
- **2D RoPE：** 把維度分成兩半，一半依 x 座標旋轉、一半依 y 座標旋轉，用來表示圖片裡的位置。
- **3D／Video RoPE：** 分成三份：時間、高、寬。

📍 Frame block 用 2D RoPE；global block 用 3D video RoPE，每個 head 的 64 維切成時間 20、高 22、寬 22（`WanRotaryPosEmbed`，借自 Wan 影片生成模型）。消融實驗裡它對 ATE 的改善最大，因為軌跡記憶裡那些只剩 6 個 token 的舊影格，要靠它才知道「是哪個時間點的」。

> 論文：RoFormer（arXiv 2104.09864）

## A5. KV cache、串流注意力與推論加速

這些是 LLM 推論的標準技術，LingBot-Map 把它們搬到影片上：

- **KV cache：**
  - 自回歸生成時，過去 token 的 K、V 不會變，存起來就不用每一步重算。
  - 代價是記憶體隨長度線性增長。
  - 📍 每張完整影格在 24 個 global 層合計約 98 MB（FP16）。
- **Sliding window attention：** 只看最近 k 個 token，記憶體固定，但會忘記更早的東西。
- **Attention sink／StreamingLLM（arXiv 2309.17453）：**
  - 發現「永遠保留最前面幾個 token + 最近的視窗」，模型就能穩定處理無限長的串流。
  - 📍 這和 LingBot-Map 的 **anchor 影格 + 滑動視窗**幾乎是同一個想法。LingBot-Map 再多加了軌跡記憶（每張舊影格留 6 個 token）。
- **PagedAttention／vLLM（arXiv 2309.06180）：**
  - 把 KV cache 像作業系統的虛擬記憶體一樣切成固定大小的 page。
  - 新增、刪除只要改 page table，不用搬動整塊記憶體。
  - 📍 LingBot-Map 一張影格的 patch token 正好佔一個 page，移出視窗就是回收一個 page。
- **FlashAttention（arXiv 2205.14135）：** 把注意力拆成小塊在 GPU 的片上記憶體裡算完，不寫出 N×N 的矩陣。數學上結果一樣，但快很多、省很多記憶體。
- **FlashInfer：** 專為 LLM 推論做的注意力 kernel 函式庫，原生支援 paged KV cache。
- **SDPA：** PyTorch 內建的 `scaled_dot_product_attention`，會自動選用 FlashAttention 等後端。
  - 📍 `--use_sdpa` 就是不用 FlashInfer、改用它。

---

## B1. 相機幾何（最重要的 3D 基礎）

- **針孔相機模型：** 3D 點 (X, Y, Z) 投影到像素：u = fx·X/Z + cx，v = fy·Y/Z + cy。
- **內參（intrinsics）K：**
  - 焦距 fx、fy 和主點 cx、cy。
  - 焦距可以換算成視角（FoV）：FoV = 2·atan(W / 2f)。
- **外參（extrinsics）：** 旋轉 R 和平移 t，描述相機在世界中的位置和朝向。
  - 要分清楚方向：world-to-camera 還是 camera-to-world（c2w）。很多 bug 都出在這裡。
- **旋轉的表示法：** 旋轉矩陣（9 個數）、四元數（4 個數，沒有萬向鎖、好插值）、軸角。
- **深度圖：** 每個像素沿光軸的距離 Z。
- **反投影（unprojection）：**
  1. 已知像素 (u, v)、深度 Z 和 K，可以求出相機座標下的 3D 點。
  2. 再用 c2w 轉到世界座標。
  3. 所有影格的點合起來，就是點雲。
- **尺度模糊：**
  - 單眼相機無法知道絕對尺度：場景放大 2 倍、相機移動也放大 2 倍，拍出來的影像一模一樣。
  - 所以重建結果只能定到「差一個比例」，評測時要先對齊尺度。
- **SE(3) 與 Sim(3)：**
  - SE(3) = 旋轉 + 平移（剛體變換）。
  - Sim(3) = 再加一個尺度。
  - 把兩段尺度不一致的重建接起來時要用 Sim(3)。

📍 Camera head 輸出 9 維：平移 3、四元數 4、FoV 2（`absT_quaR_FoV`）。點雲 = 深度 + 位姿反投影。Windowed 模式用 Sim(3) 接合視窗。Anchor 影格的作用就是解決尺度模糊：先定一個尺度，後面都照它。

## B2. 傳統管線：SfM、SLAM、BA

- **SfM（Structure from Motion）：**
  - 離線處理一堆無序照片。
  - 流程：抽特徵點、跨圖配對、估相機、三角化、最佳化。代表工具是 COLMAP。
- **SLAM（同時定位與建圖）：** 線上處理影片串流。典型模組：
  - **Tracking：** 每張新影格對齊到地圖上，估出位姿。
  - **Local mapping：** 對最近的 keyframe 做局部最佳化。
  - **Loop closure：** 發現「回到以前來過的地方」，修正整條軌跡。
  - **Keyframe：** 不是每張影格都存，只挑有代表性的。
- **Bundle Adjustment（BA）：** 同時最佳化所有相機位姿和 3D 點，讓重投影誤差最小。準確但很慢，是傳統方法的核心。
- **Visual Odometry（VO）：** 只做連續影格之間的相對運動估計，不做全域修正，所以誤差會一直累積（drift）。
- **Drift：** 每一步的小誤差累積起來，軌跡越走越歪。長序列的頭號敵人。
- **MVS（Multi-View Stereo）：** 相機已知時，做稠密深度和重建。

📍 LingBot-Map 的三種上下文直接對應 SLAM 的分工：anchor 對應參考座標系，視窗對應局部地圖，軌跡記憶對應全域地圖。差別是全部用注意力學出來，不做 BA。論文自承「沒有 loop closure」，就是指 SLAM 的那個模組。

> 代表作：ORB-SLAM（arXiv 1502.00956）、DROID-SLAM（arXiv 2108.10869，可微分的 BA）

## B3. 學習式 3D 重建的系譜

| 模型 | 年份 | 關鍵想法 |
|---|---|---|
| DPT | 2021 | 用 ViT 做稠密預測（深度、分割）。把不同層的 token 重組成影像狀的特徵圖，逐層融合、上採樣 |
| DUSt3R | 2023 | 兩張圖直接回歸 **pointmap**（每個像素對應的 3D 座標），不需要相機參數。多於兩張要靠額外的全域最佳化拼起來 |
| MASt3R | 2024 | DUSt3R 加上特徵匹配，配對更準 |
| VGGT | 2025 | 任意張數一次前饋：**frame／global 交替注意力**、camera token、反覆精修的 camera head、DPT head。全部影格互相看（非串流） |
| CUT3R | 2025 | 串流化：用 RNN 式的持久狀態吸收每張新影格。狀態被壓得太緊，容易遺忘 |
| StreamVGGT／Stream3R | 2025 | 把 VGGT 改成因果注意力 + KV cache。保留全部歷史，記憶體線性增長 |
| **LingBot-Map** | 2026 | VGGT 的因果版 + 有選擇地保留歷史（anchor／視窗／6 token 軌跡記憶） |

- **Frame／global 交替注意力**（VGGT 的核心）：
  - Frame 層只在單張圖內看，負責理解這張圖。
  - Global 層跨所有圖看，負責建立影格之間的對應關係。
  - 兩種層交替疊 24 次。
- **反覆精修（iterative refinement）：** camera head 不是一次輸出位姿，而是跑好幾輪，每輪在上一輪的結果上加修正量。概念來自 RAFT 光流。

📍 LingBot-Map 的骨架幾乎就是 VGGT。要讀懂它，最直接的捷徑是先讀 VGGT（arXiv 2503.11651），再看 LingBot-Map 把 global attention 改成什麼樣子。

> DPT（arXiv 2103.13413）、DUSt3R（arXiv 2312.14132）、CUT3R（arXiv 2501.12387）

## B4. 評測指標

| 指標 | 量什麼 | 怎麼算 |
|---|---|---|
| **ATE**（Absolute Trajectory Error） | 整條軌跡的全域準確度 | 先用 Umeyama 把預測軌跡做 Sim(3) 對齊到真值，再算位置誤差的 RMSE（公尺） |
| **RPE**（Relative Pose Error） | 局部、短距離的相對運動準確度 | 相鄰影格的相對位姿誤差，分平移（trans）和旋轉（rot） |
| **AUC@θ** | 位姿準確率 | 任兩張影格的相對旋轉／平移角度誤差，統計「誤差 < θ 度」的比例，再對門檻積分。越高越好 |
| **Accuracy／Completeness／F1** | 點雲品質 | Accuracy：預測點到真值的距離；Completeness：真值點有沒有被覆蓋；F1 是兩者的綜合 |

- ATE 低、RPE 高：全域形狀對，但局部抖動。
- ATE 高、RPE 低：每一步都算準，但累積 drift。

## B5. 訓練相關概念

- **Relative pose loss：** 除了監督每張影格的絕對位姿，也監督視窗內兩兩影格的相對位姿。論文消融：拿掉後旋轉誤差惡化 2.4 倍。
- **尺度正規化：** 真值點雲除以 anchor 影格點雲到原點的平均距離，讓不同資料集的尺度一致。
- **漸進式課程（progressive curriculum）：** 先訓練短序列，再逐步加長（24 → 320 張影格）。直接訓練長序列的話，早期的位姿錯誤會一路傳下去，很難收斂。
- **Context parallelism：** 序列太長、一張 GPU 放不下時，把不同影格分到不同 GPU，用 all-to-all 通訊算注意力。

---

## 建議閱讀順序

1. **只想看懂 LingBot-Map：** A1 → A2 → B1 → B3 的 VGGT 段落 → A5 的 KV cache 和 StreamingLLM → 回頭讀 LingBot-Map。
2. **想打好 3D 視覺基礎：** B1 → B2 → DUSt3R → VGGT → CUT3R → LingBot-Map。
3. **想搞懂推論加速：** A5 全部：FlashAttention → PagedAttention → StreamingLLM → FlashInfer。

## 討論紀錄

<!-- 每次討論一個小節：### YYYY-MM-DD：主題 -->
