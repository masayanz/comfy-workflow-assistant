$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$SettingsPath = Join-Path $ProjectRoot 'data\settings.json'
$Port = 7865
$OpenBrowser = $true

if (Test-Path -LiteralPath $SettingsPath) {
    try {
        $SavedSettings = Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
        if ($SavedSettings.port -ge 1024 -and $SavedSettings.port -le 65535) { $Port = [int]$SavedSettings.port }
        if ($null -ne $SavedSettings.open_browser) { $OpenBrowser = [bool]$SavedSettings.open_browser }
    } catch {
        Write-Warning '設定ファイルを読み取れません。既定のポート7865で起動します。'
    }
}

py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 12))"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12以上が必要です。py -3 --version でバージョンを確認してください。' }

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host 'Python仮想環境を作成しています...'
    py -3 -m venv (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12が見つかりません。Python 3.12以上をインストールしてください。' }
}

& $Python -c "import sys; raise SystemExit(sys.version_info < (3, 12))"
if ($LASTEXITCODE -ne 0) { throw '既存のPython仮想環境がPython 3.12未満です。.venvを作り直してください。' }

Write-Host '依存関係を確認しています...'
& $Python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw '依存関係のインストールに失敗しました。' }

if ($OpenBrowser) {
    $Url = "http://127.0.0.1:$Port"
    $BrowserScript = "for (`$Attempt = 0; `$Attempt -lt 120; `$Attempt++) { try { Invoke-WebRequest -Uri '$Url' -UseBasicParsing -TimeoutSec 1 | Out-Null; Start-Process '$Url'; exit } catch { Start-Sleep -Seconds 1 } }"
    $EncodedBrowserScript = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($BrowserScript))
    Start-Process 'powershell.exe' -WindowStyle Hidden -ArgumentList @('-NoProfile', '-EncodedCommand', $EncodedBrowserScript)
}
Write-Host "Comfy Workflow Builderを起動します: http://127.0.0.1:$Port"
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
