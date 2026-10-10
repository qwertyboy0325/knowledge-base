# 知識庫維護規則

這是使用者的個人知識庫。使用者給一個論文、repo、文章網址或主題，要你「加進知識庫」時：

1. **實際讀原始資料**：clone repo、讀 README 與核心程式碼、讀論文 PDF（`pdftotext`）。不要只憑記憶寫，數字要來自原文。
2. **選資料夾**：論文（含附程式碼的論文）→ `papers/`；開源專案／工具 → `repos/`；其他 → `notes/`。
3. **檔名**：小寫加連字號的短名稱，例如 `papers/lingbot-map.md`。先檢查是否已有同主題的條目，有就更新而不是重複新增。
4. **內容**：照 `templates/entry.md` 的段落寫，以繁體中文為主，專有名詞保留英文。front matter 的 `title`、`date`（今天）、`tags`、`summary` 必填；`status` 預設 `to-read`，除非使用者說讀過了。
5. **標籤**：優先沿用 `INDEX.md`「依標籤」裡已存在的標籤；全小寫、用連字號。
6. **重建索引**：`python scripts/build_index.py`，確認沒有 error。
7. Commit 訊息格式：`add: <標題>` 或 `update: <標題>`。

`INDEX.md` 是產生出來的檔案，不要手動編輯。
