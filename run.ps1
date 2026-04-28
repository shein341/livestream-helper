param(
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Fail($Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Read-DotEnv($Path) {
    $result = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $result
    }
    foreach ($raw in Get-Content -LiteralPath $Path) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }
        $idx = $line.IndexOf("=")
        $key = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        $result[$key.Trim()] = $value.Trim().Trim('"').Trim("'")
    }
    return $result
}

function Require-Command($Name, $InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail "$Name is not installed or not in PATH. $InstallHint"
    }
}

Require-Command "docker" "Install Docker Desktop first."
Require-Command "ollama" "Install Ollama first: https://ollama.com/download"

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Fail "Docker is not running."
}

docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    Fail "Docker Compose is not available. Update Docker Desktop."
}

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 10 | Out-Null
} catch {
    Fail "Ollama is not running at http://127.0.0.1:11434. Start Ollama and retry."
}

if (-not (Test-Path -LiteralPath ".env")) {
    if (-not (Test-Path -LiteralPath ".env.example")) {
        Fail ".env.example is missing."
    }
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Fail ".env was created from .env.example. Fill RAG_ANSWER_API_KEY, then rerun .\run.ps1."
}

$envValues = Read-DotEnv ".env"
$apiKey = $envValues["RAG_ANSWER_API_KEY"]
if (-not $apiKey -or $apiKey -eq "your_answer_api_key_here") {
    Fail "RAG_ANSWER_API_KEY is missing in .env. Fill it with your answer model API key, then rerun .\run.ps1."
}

$models = @()
$rewriteModel = $envValues["RAG_QUERY_REWRITE_MODEL"]
$embeddingModel = $envValues["RAG_EMBEDDING_MODEL"]
if (-not $rewriteModel) { $rewriteModel = "qwen3.5:4b" }
if (-not $embeddingModel) { $embeddingModel = "bge-m3:latest" }
$models += $rewriteModel
$models += $embeddingModel

$tagResponse = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 10
$installed = @{}
foreach ($model in $tagResponse.models) {
    $installed[$model.name] = $true
}

foreach ($model in ($models | Select-Object -Unique)) {
    if (-not $installed.ContainsKey($model)) {
        Write-Host "Pulling Ollama model: $model" -ForegroundColor Yellow
        ollama pull $model
        if ($LASTEXITCODE -ne 0) {
            Fail "Failed to pull Ollama model: $model"
        }
    } else {
        Write-Host "Ollama model ready: $model" -ForegroundColor Green
    }
}

$composeArgs = @("compose", "up")
if (-not $NoBuild) {
    $composeArgs += "--build"
}

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Green
Write-Host "App:     http://localhost:8000" -ForegroundColor Cyan
Write-Host "Swagger: http://localhost:8000/swagger" -ForegroundColor Cyan
Write-Host ""

docker @composeArgs
