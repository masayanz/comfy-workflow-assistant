# Comfy Workflow Builder

ComfyUI初心者から初級・中級ユーザー向けの、ローカルで動作するワークフロー作成ツールです。モデルと生成条件を選び、ComfyUI API形式での直接実行と、キャンバスで編集できるWorkflow JSONの保存ができます。

## MVPの対応範囲

- Windows 11を想定した起動スクリプト
- ComfyUI APIの接続状態表示
- Checkpoint、LoRA、VAE、ControlNetなどのモデル一覧取得と系統分類
- Upscale Modelの選択には、親機ComfyUI APIが認識するモデル一覧を使用
- ComfyUIのファイルシステムにアクセスできない場合は `/object_info` から利用可能モデル一覧を取得
- ファイル名・保存先を使ったSD1.5 / SDXL / Fluxの簡易分類
- 推定できなかったモデル系統の手動修正と保存
- SD1.5 / SDXL txt2img、SDXL img2img、画像アップスケールのWorkflow生成、JSON保存、ComfyUI実行、生成画像表示
- 上記WorkflowのComfyUIキャンバス用JSON保存（API Promptとは別形式）
- 既存のComfyUI UI Workflow / API Prompt JSONの読み込みと読み取り専用解析
- 既存UI WorkflowのCheckpoint / LoRA / 生成設定 / 解像度の部分編集、差分確認、別名保存、明示Queue実行
- 1つまでのLoRA選択と適用強度の調整
- `model_profiles/` で管理するモデル別デフォルト値と対応機能
- ComfyUIルートフォルダとAPI URLの設定

現在、SD1.5 / SDXL txt2img、SDXL img2img、標準の`LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage`によるアップスケールWorkflowの生成と実行に対応しています。アップスケールの実行には親機ComfyUIが認識するUpscale Modelが必要です。LoRAはSD1.5 / SDXLで1つ適用できます。Flux txt2imgの構成生成は対応していますが、実機モデル不足のため実生成は未確認です。FluxのLoRA/img2img、SD1.5 img2img、複数LoRA、生成履歴は未対応です。モデル分類はファイル名と保存先による推定なので、判定が「不明」になることがあります。

アップスケール用モデルはアプリを動かすPC上のファイルではなく、接続先の親機ComfyUIが`/models/upscale_models`で認識したものだけを使います。Web UIの「親機ComfyUI診断」ではComfyUIのバージョン、GPU、実際のモデル保存先、モデル在庫、必要ノード、登録済みCustom Nodeを確認できます。親機にUpscale Modelがない場合、診断画面に表示されるパスへ配置してください。Stability Matrixの親機向けに、SHA-256を確認し、同名ファイルを上書きしない導入スクリプトを`scripts/Install-RealESRGAN-On-Parent.ps1`に用意しています。親機が別PCの場合はスクリプトを親機へコピーして、親機のPowerShellから手動で実行します。

Workflow生成時は、ComfyUIキャンバスに読み込んで編集する `.workflow.json` と、ComfyUI APIへ送信する `.api.json` を別々に保存できます。どちらも`generated_workflows/`に保存され、ブラウザにもダウンロードされます。

既存JSONはWeb UIの「既存Workflowを解析」から読み込めます。形式、ノード、リンク、モデル、生成パラメータ、処理概要を表示し、親ComfyUIの`/object_info`とモデル一覧に照合します。解析は読み取り専用です。「このWorkflowを編集」を押した場合だけブラウザー内に編集用コピーを作り、Checkpoint、既存LoRA、Seed、Steps、CFG、Sampler、Scheduler、img2imgのDenoise、既知のLatent解像度を部分patchできます。LoRA Loaderの削除では既知のMODEL / CLIP接続をバイパスします。原本は上書きせず、`*.edited.workflow.json`としてダウンロードします。Custom Nodeや対象外ノードのデータは保持しますが、編集・API変換できないWorkflowはQueue実行できません。Queue送信は明示ボタンと確認ダイアログの後に限ります。親機に接続できない場合はモデルとノードを照合不明とし、Sampler / Schedulerの選択肢を取得できない場合はその項目を編集できません。RTX 3060 12GB向け注意は解像度やノード構成からの目安で、VRAM使用量の数値予測ではありません。

SD1.5、SDXL、Fluxの解像度・Steps・CFG・Sampler・Schedulerの既定値は `model_profiles/` のJSONで管理します。Flux ProfileはUNET、2つのText Encoder、VAEを個別に選択します。必要なFluxモデルがComfyUIにない場合もAPI PromptとUI Workflowの未設定プレビューを作れますが、保存とQueue実行は必要なモデルが揃うまで無効です。CheckpointまたはProfileを選ぶとBackendから設定を読み込み、画面へ反映します。

SDXL img2imgとアップスケールではPNG / JPEG / WEBP（20MB以下）の入力画像を1枚アップロードします。画像本体はComfyUIの`input/`直下へUUID名で送り、WorkflowはComfyUIの`LoadImage`から参照します。アップスケールの出力サイズは生成後の画像から表示します。事前の倍率はモデル名から推定できる場合のみ参考表示します。ローカル接続でComfyUIルートを確認できる場合、アプリ画面を開いた際または次回アップロード時に、アプリ専用の命名規則に合う30日以上前の画像だけを削除します。リモートComfyUIには標準の画像削除APIがないため、期限を過ぎたファイルもComfyUI側に残ります。不要な画像はComfyUIのinputフォルダから整理してください。

## 必要環境

- Windows 11
- Python 3.12以上（Python Launcherの `py` コマンドが利用可能）
- インストール済みのComfyUIと、利用するSD1.5またはSDXL Checkpoint

## 起動

リポジトリのルートでPowerShellを開き、次を実行します。

```powershell
.\start.ps1
```

初回起動時に `.venv` を作成し、`requirements.txt` の依存関係をインストールして、`http://127.0.0.1:7865` でWeb UIを起動します。サーバーはlocalhostのみにbindします。

ComfyUIは既定で `http://127.0.0.1:8188` に接続します。自動検出されない場合は画面右上の設定からComfyUI API URLと、`models` フォルダを含むComfyUIルートフォルダを指定してください。Stability Matrix環境でもComfyUIパッケージのルートを指定できます。

## 使い方

1. 左側でSD1.5またはSDXL Checkpointを選びます。モデル系統が不明な場合は手動で分類できます。
2. 生成したい内容を入力し、解像度やSamplerなどを指定します。
3. 「Workflow生成」でキャンバス用Workflow JSONとAPI Prompt JSONを作成し、構成を確認します。
4. 「ComfyUI Workflowを保存」または「API Promptを保存」で形式を選んでダウンロードします。サーバー側にも `generated_workflows/` へ保存します。
5. キャンバスで編集する場合は、保存した`.workflow.json`をComfyUIのLoadから開きます。アプリから直接生成する場合は「ComfyUIで生成」をクリックします。完了すると生成画像と設定が画面に表示されます。

既存Workflowを確認する場合は、プレビュー欄の「ComfyUI Workflow JSON」からファイルを選んで「Workflowを読み込んで解析」をクリックします。対応形式はComfyUI UI Workflow JSONとAPI Prompt JSONです。結果の「ComfyUIを開く」からComfyUIを開き、編集する場合はキャンバスのLoadから元JSONを選んでください。

既存UI Workflowをアプリ内で部分編集する場合は、解析結果の「このWorkflowを編集」をクリックします。値を変更したら「変更を適用して差分を確認」で差分とValidationを確認し、`新しいWorkflowとして保存`で別ファイルへ出力します。Queue実行は`ComfyUIでテスト生成`を押し、確認ダイアログで送信を承認します。Checkpoint / LoRAの在庫、互換性、Sampler / Scheduler候補は親ComfyUIから取得します。API Prompt JSONは解析のみで、キャンバスWorkflowとしての編集には対応していません。

Image to Imageを選び、画像をアップロードすると、入力画像のサイズをそのまま使います。「変化の強さ」は`denoise`（0.0〜1.0）で、低いほど元画像に近い結果になります。

ComfyUI側に各Workflowで使うノードが必要です。Upscaleでは`LoadImage`、`UpscaleModelLoader`、`ImageUpscaleWithModel`、`SaveImage`を確認します。独自ノードを使うWorkflowは解析で一覧表示できますが、アプリからの実行・編集は未対応です。

`.api.json`はComfyUI APIへ送るPrompt形式、`.workflow.json`はComfyUIのグラフ画面で読み込んで編集する形式です。2種類のJSONはWorkflowDefinitionから別々に生成します。

## 設定とログ

- 設定ファイル: `data/settings.json`（初回保存時に作成）
- ログ: `logs/app.log`
- 保存workflow: `generated_workflows/`

Web UIポートと起動時のブラウザ表示も設定できます。ポート変更は設定保存後にアプリを再起動すると反映されます。

これらのローカル生成ファイルはGit管理対象外です。

## API

- `GET /api/status`
- `GET /api/comfy/status`
- `GET /api/comfy/models`（親機ComfyUIの認識一覧）
- `GET /api/diagnostics`（親機環境・モデル・ノード診断）
- `GET /api/models` / `POST /api/models/scan`（既存のモデルスキャン）
- `GET /api/loras`
- `GET /api/model-profiles` / `GET /api/model-profiles/{id}`
- `GET /api/settings` / `PUT /api/settings`
- `POST /api/workflow/build`
- `POST /api/workflow/import`（multipartの`file`でJSONを受け取り、実行せず解析）
- `POST /api/workflow/edit/prepare` / `POST /api/workflow/edit/apply`（部分patchと差分・Validation）
- `POST /api/workflow/edit/queue`（明示的な編集Workflow Queue送信）
- `POST /api/uploads/image`
- `POST /api/workflow/save`
- `POST /api/workflow/save-ui`
- `POST /api/workflow/run`

## 開発

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 7865
```

テストは `python -m unittest discover -s tests -v` で実行できます。

## Contributing

不具合報告や改善提案、Pull Requestを歓迎します。再現手順、ComfyUIのバージョン、必要なモデルやノードを添えてください。モデルファイルや個人環境の設定をPull Requestへ含めないでください。

## License

MIT License。詳細は [LICENSE](LICENSE) を参照してください。
