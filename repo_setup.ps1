Write-Host "`n--- Updating Repository Structure ---`n"

# 1. Create folders
New-Item -ItemType Directory -Force -Name "assets" | Out-Null
New-Item -ItemType Directory -Force -Name "docs" | Out-Null
New-Item -ItemType Directory -Force -Name "notebooks" | Out-Null
New-Item -ItemType Directory -Force -Name "tests" | Out-Null

# 2. Move images into assets
Get-ChildItem -File *.png, *.jpg, *.jpeg | Move-Item -Destination "assets" -Force

# 3. Generate simple README
$readme = @"
# Stock Price Prediction (LSTM + Streamlit)

A real-time stock analysis tool with:
- 10+ years historical data
- RSI, Volatility, Daily Returns
- LSTM price prediction
- Buy / Hold / Avoid recommendation
- CSV download support
"@

Set-Content -Path "README.md" -Value $readme

# 4. Git update
git add .
git commit -m "Repo restructure + auto README"
git push

Write-Host "`n--- Done ---`n"
