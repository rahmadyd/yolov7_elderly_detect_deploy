param(
  [string]$CondaEnv = "de_master"
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RootDir "backend\api"
$FrontendDir = Join-Path $RootDir "frontend"

if ($CondaEnv.Trim()) {
  $BackendCommand = "conda activate $CondaEnv; cd `"$BackendDir`"; uvicorn main:app --reload --port 8000"
} else {
  $BackendCommand = "cd `"$BackendDir`"; uvicorn main:app --reload --port 8000"
}

$FrontendCommand = "cd `"$FrontendDir`"; npm.cmd run dev"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCommand
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCommand

Write-Host ""
Write-Host "Backend  : http://localhost:8000"
Write-Host "Frontend : http://localhost:5173"
Write-Host ""
Write-Host "Tutup dua window PowerShell yang terbuka kalau mau stop server."
