# Knowledge Base

個人知識庫：看到有興趣的論文、開源專案、文章或想法，就整理成一篇 Markdown 放進來，索引會自動產生。

👉 **[瀏覽索引 INDEX.md](INDEX.md)**

## 結構

| 資料夾 | 放什麼 |
|---|---|
| `papers/` | 論文（含附帶程式碼的論文） |
| `repos/` | 開源專案、工具、函式庫 |
| `notes/` | 文章、概念、雜記 |
| `templates/entry.md` | 新條目範本 |
| `scripts/build_index.py` | 從每篇的 front matter 產生 `INDEX.md` |

## 新增一筆

1. 複製範本：`cp templates/entry.md papers/<短名稱>.md`（檔名用小寫加連字號）
2. 填好開頭的 front matter：`title`、`date`、`tags`、`summary` 必填，`type`、`status`、`source` 選填
3. 重建索引：`python scripts/build_index.py`
4. Commit + push

忘了跑第 3 步也沒關係：push 到 `main` 後 GitHub Action 會自動重建並 commit `INDEX.md`。

也可以直接叫 Claude Code 幫忙：「把 <網址> 研究一下加進知識庫」，規則寫在 [CLAUDE.md](CLAUDE.md)。

## 標籤慣例

- 全小寫、用連字號：`3d-reconstruction`、`slam`、`llm-inference`
- 先沿用 [INDEX.md](INDEX.md)「依標籤」裡已有的標籤，真的沒有再新增
