---
title: 串流系統的共通底層：3D 重建 ↔ 即時音樂協作
type: note
date: 2026-10-10
status: reading
tags: [streaming, slam, kv-cache, clock-sync, echlub, prerequisites]
summary: 把 LingBot-Map 背後的底層概念（drift、尺度模糊、對齊後評測、快取記憶體管理、因果處理）對應到 EchLub 的時鐘同步、效能證據與狀態管理
---

# 串流系統的共通底層：3D 重建 ↔ 即時音樂協作

[LingBot-Map](../papers/lingbot-map.md) 的模型本身和 EchLub 無關，但[它背後的底層知識](3d-reconstruction-prerequisites.md)有不少是「串流系統」共通的。這篇把它們對應到 EchLub，附上 EchLub 程式碼裡的實際位置。

**最重要的發現：SLAM 的「狀態估計」思維和 EchLub 的時鐘同步幾乎是同一個數學問題。**

## 1. 時鐘同步 = 一維的位姿估計（關聯最強）

EchLub 現況（`crates/echlub-performance/src/live_clock.rs`、`live_derived.rs`）：
- 每次 probe 用 NTP 式四個時間戳算 `offset = ((t1−t0) + (t2−t3)) / 2` 和 RTT。
- 整段 session 的 offset 取**中位數**，離散程度用 MAD。

對應關係：

| 3D 重建 | 時鐘同步 |
|---|---|
| 相機位姿（R, t） | 對方時鐘相對我方的 offset |
| Sim(3) 的尺度 s | 時鐘**速率比**（skew）。兩台機器的晶振不會完全一樣快，例如 48000 Hz 對 48003 Hz |
| Drift（誤差隨時間累積） | 時鐘漂移：offset 隨時間線性變化 |
| Umeyama 對齊（解 s, R, t） | 對 (本地時間, 遠端時間) 做線性回歸，解出 offset 和速率 |
| 只挑高品質的 keyframe | 只用 RTT 最小的 probe：排隊延遲最少、最可信 |
| 單眼的尺度模糊（原理上不可觀測） | **路徑不對稱不可觀測**：只量來回時間，無法分辨「時鐘差」和「去程、回程走得不一樣快」 |

**對 EchLub 的具體啟發：**
- **用中位數合併 offset，等於假設整段 session 的 offset 是常數（沒有 skew）。**
  - 一般晶振誤差約 10–100 ppm，一分鐘可能漂 0.6–6 ms。實際上作業系統若有 NTP 校時，可能較小，也可能出現校時造成的跳動。
  - 如果 session 夠長，中位數會落在中段的值，MAD 也會被漂移撐大，看起來像「量測很不穩」。
  - 值得做的一步：把 offset 對時間作圖，或做線性擬合，看看斜率是不是顯著不為 0。這比較的是「常數模型」和「offset + 速率模型」。
- **路徑不對稱屬於原理上的限制，不是實作問題。**
  - `live-two-peer-known-issues.md` 已寫「route asymmetry affects it」。
  - 可以更精確地說：這和單眼 SLAM 的尺度一樣，是原理上不可觀測的量。只用 RTT 的方法，最多只能把誤差界定在 ±RTT/2 之內。要突破只能引入外部參考，例如 GPS 時鐘，或聲學的 mouth-to-ear 量測。
  - 這正好支撐 EchLub「不宣稱 one-way latency」的邊界。

## 2. 「先對齊，再評測」與 ATE／RPE 的分工

- 3D 重建算 ATE 之前一定要先做 Sim(3) 對齊，否則量到的是「座標系不同」，而不是「重建不準」。
- **EchLub 對應：** 跨裝置的時間戳在比較前，必須先扣掉 offset（如有需要也校正 skew）。否則延遲數字裡混進了時鐘差。
- **ATE 與 RPE 的分工可以直接借用：**
  - 「絕對同步誤差」：整段 session 兩邊時間軸對得多準，類似 ATE。
  - 「相對抖動」：相鄰事件之間的時間誤差，也就是 jitter，類似 RPE。
  - 兩者分開報告，才看得出是「整體偏移」還是「局部不穩」。EchLub 現在已有 `jitter_ms`。
- **參數網格式的呈現：** LingBot-Map 論文表 9 把「設定 → 精度／各硬體速度」排成網格。EchLub 的效能證據若要比較 buffer 大小、傳輸方式、瀏覽器，可以用同樣格式讓取捨一目了然。

## 3. 因果處理：只能看過去，看未來就要付延遲

- LingBot-Map 是因果的：每張影格只看現在和過去。
- 即時音訊也是：任何「往後看」（lookahead、更大的 jitter buffer、平滑濾波）都直接換算成延遲。
- 共通的設計問題：要多少「過去」才夠準，要多少「未來」才值得等。
  - LingBot-Map 的答案是視窗 48–64 之後精度就飽和了。
  - EchLub 對應的是 jitter buffer 的大小和延遲預算之間的取捨，這需要量測，不能用猜的。

## 4. 記憶體管理：PagedAttention 的教訓 = 即時音訊的鐵律

- LingBot-Map 用 paged KV cache 的原因：連續記憶體版本每次增刪都要重新配置記憶體，速度只有約一半（10.5 對 20 FPS）。
- 改成固定大小的 page、預先配置好的 pool、只改 page table，就不會在熱路徑上配置記憶體。
- **即時音訊有同樣的鐵律：** audio callback 裡不能 malloc、不能上鎖，要用預先配置的 ring buffer 和 object pool。
- **對 EchLub 的意義：** 如果未來真的走向 Rust 音訊核心（目前 README 明確說「未選定」），這是第一天就要遵守的設計約束。

## 5. 有界的歷史：基準 + 最近細節 + 舊摘要

- LingBot-Map／StreamingLLM 的結構：固定基準（anchor），加上最近幾張的完整細節（視窗），再加上更早歷史的精簡摘要（6 token）。
- **EchLub 對應：** Rust 核心有 document ops 的 replay，加上狀態雜湊。如果操作紀錄成長到重播變慢：
  - 快照當基準，保留最近的完整 ops，更早的只留雜湊鏈作為完整性證明。
  - **後加入的 peer** 也適用：給它快照 + 最近的 ops，而不是整段歷史。
- 注意：這就是 event sourcing 的快照／壓縮模式。依 EchLub 的原則，應等量測到瓶頸再做。

## 6. 不太相關的部分

- Attention、ViT、DINOv2、RoPE、DPT：只有走「AI 生成音樂」方向才用得到。音樂生成模型同樣用 transformer + RoPE + KV cache 做串流生成。目前 EchLub 明確沒選這個方向，先放著。
- 相機幾何的細節（內參、反投影）：沒有對應。

## 優先順序建議（如果要做）

1. **最便宜、最有價值：** 把現有的 clock probe offset 對時間作圖，確認有沒有 skew。現有資料就能做，不用寫新功能。
2. 在效能證據裡分開報告「絕對同步誤差」與「jitter」，並把 asymmetry 限制寫成 ±RTT/2 的明確界限。
3. 其餘（快照壓縮、ring buffer 設計）等對應的方向或瓶頸真的出現再說。

## 討論紀錄

<!-- 每次討論一個小節：### YYYY-MM-DD：主題 -->
