param(
    [Parameter(Mandatory)][ValidateSet("nb01","nb02","nb03","nb04")]
    [string]$nb
)

$map = @{
    "nb01" = @{ file = "NB01_verify_data.ipynb";  slug = "harish77718/nb01-verify-data"  }
    "nb02" = @{ file = "NB02_potsdam_eval.ipynb"; slug = "harish77718/nb02-potsdam-eval" }
    "nb03" = @{ file = "NB03_hessen_infer.ipynb"; slug = "harish77718/nb03-hessen-infer" }
    "nb04" = @{ file = "NB04_demo.ipynb";         slug = "harish77718/nb04-demo"         }
}

$notebook = $map[$nb].file
$slug     = $map[$nb].slug
$pushDir  = "notebooks\push\$nb"

# Kaggle CLI v2 inconsistently reads credentials.json for kernel ops.
# Explicitly set the token so all operations use the same auth path.
$credFile = "$env:USERPROFILE\.kaggle\credentials.json"
if (Test-Path $credFile) {
    $env:KAGGLE_API_TOKEN = (Get-Content $credFile | ConvertFrom-Json).access_token
} else {
    Write-Host "No credentials.json found — run: kaggle auth login" -ForegroundColor Red
    exit 1
}

Write-Host "Copying $notebook -> $pushDir\" -ForegroundColor Cyan
Copy-Item "notebooks\$notebook" "$pushDir\$notebook" -Force

Write-Host "Pushing to Kaggle..." -ForegroundColor Cyan
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
kaggle kernels push -p $pushDir

Write-Host ""
Write-Host "Check status:  kaggle kernels status $slug" -ForegroundColor Yellow
Write-Host "Pull outputs:  kaggle kernels output $slug -p results\" -ForegroundColor Yellow
