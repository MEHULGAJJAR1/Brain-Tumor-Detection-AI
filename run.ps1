$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (Test-Path '.env') {
    Get-Content '.env' | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $parts = $line -split '=', 2
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
        }
    }
}

python app.py
