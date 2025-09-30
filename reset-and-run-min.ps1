# Minimal reset + install + dev (ASCII only)
$ErrorActionPreference = 'Stop'

# Go to script directory (project root)
Set-Location -Path $PSScriptRoot

if (-not (Test-Path '.\package.json')) {
  Write-Host 'ERROR: package.json not found. Place this script in project root.'
  exit 1
}

# Stop any leftover node
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Clean build outputs and deps
try {
  npx rimraf node_modules .next
} catch {
  Remove-Item -Recurse -Force node_modules, .next -ErrorAction SilentlyContinue
}

# Verify cache (optional)
try { npm cache verify | Out-Null } catch {}

# Install and run
npm install
npm run dev