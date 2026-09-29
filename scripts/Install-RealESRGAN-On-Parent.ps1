param(
    [string]$ComfyRoot = 'E:\StabilityMatrix\Data\Packages\ComfyUI'
)

$ErrorActionPreference = 'Stop'

$ModelFileName = 'RealESRGAN_x4plus.pth'
$ModelUri = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
$ExpectedSha256 = '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'

if (-not (Test-Path -LiteralPath $ComfyRoot -PathType Container)) {
    throw "指定した親機ComfyUIフォルダが見つかりません: $ComfyRoot"
}

$TargetDirectory = Join-Path $ComfyRoot 'models\upscale_models'
$TargetPath = Join-Path $TargetDirectory $ModelFileName
$DataRoot = Split-Path (Split-Path $ComfyRoot -Parent) -Parent
$KnownUpscaleDirectories = @(
    $TargetDirectory,
    (Join-Path $DataRoot 'Models\ESRGAN'),
    (Join-Path $DataRoot 'Models\RealESRGAN'),
    (Join-Path $DataRoot 'Models\SwinIR')
)

foreach ($Directory in $KnownUpscaleDirectories) {
    $ExistingPath = Join-Path $Directory $ModelFileName
    if (Test-Path -LiteralPath $ExistingPath -PathType Leaf) {
        $ExistingHash = (Get-FileHash -LiteralPath $ExistingPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ExistingHash -eq $ExpectedSha256) {
            Write-Output "同一モデルがすでにあります: $ExistingPath"
            Write-Output 'ComfyUIを再起動するか、アプリのモデル再スキャンを実行してください。'
            exit 0
        }
        throw "同名の異なるファイルがあるため中断しました。既存ファイルは変更していません: $ExistingPath"
    }
}

New-Item -ItemType Directory -Path $TargetDirectory -Force | Out-Null
$TemporaryPath = Join-Path $TargetDirectory ('.' + $ModelFileName + '.' + [guid]::NewGuid().ToString('N') + '.partial')
try {
    Write-Output "公式配布元からダウンロードします: $ModelUri"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $ModelUri -OutFile $TemporaryPath -UseBasicParsing
    $DownloadedFile = Get-Item -LiteralPath $TemporaryPath
    if ($DownloadedFile.Length -lt 50MB) {
        throw 'ダウンロードファイルが想定より小さいため中断しました。'
    }
    $DownloadedHash = (Get-FileHash -LiteralPath $TemporaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $ExpectedSha256) {
        throw "SHA-256が一致しません。期待値=$ExpectedSha256 実際=$DownloadedHash"
    }
    [System.IO.File]::Move($TemporaryPath, $TargetPath)
    Write-Output "親機ComfyUIへ配置しました: $TargetPath"
    Write-Output 'ComfyUIを再起動し、アプリで「再スキャン」してください。'
}
finally {
    if (Test-Path -LiteralPath $TemporaryPath -PathType Leaf) {
        Remove-Item -LiteralPath $TemporaryPath -Force
    }
}
