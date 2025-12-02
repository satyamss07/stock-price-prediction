Write-Host "`n=== Organizing Project (Clean & Fixed) ===`n"

# -----------------------------
# 1. Ensure assets folder exists
# -----------------------------
if (!(Test-Path -Path "assets")) {
    New-Item -ItemType Directory -Path "assets" | Out-Null
    Write-Host "Created assets/ folder"
}

# -----------------------------
# 2. Move all screenshots into assets/
# -----------------------------
$images = Get-ChildItem -File -Include *.png, *.jpg, *.jpeg

foreach ($img in $images) {
    Move-Item -Path $img.FullName -Destination "assets" -Force
}

Write-Host "✔ Moved screenshots into assets/"

# -----------------------------
# 3. Generate README header safely
# -----------------------------
$header = @'
# Stock Price Prediction (LSTM + Streamlit)

![Banner](assets/banner.png)

A fully interactive real-time stock analysis and prediction web app.

---
'@

# -----------------------------
# 4. Auto-generate screenshot section
# -----------------------------
$ss_section = "## Screenshots`n`n"

$ss_files = Get-ChildItem assets | Where-Object { $_.Name -match "png|jpg|jpeg" }

foreach ($f in $ss_files) {
    $ss_section += "![Screenshot](assets/$($f.Name))`n`n"
}

# -----------------------------
# 5. Full README text
# -----------------------------
$readme = @"
$header

## 📌 Features  
- Real-time stock data (Yahoo Finance)  
- Supports custom date ranges (up to 10 years)  
- Automatic indicators: SMA, RSI, Volatility  
- LSTM deep learning model  
- Buy / Hold / Avoid recommendation engine  
- Download predictions as CSV  
- Rich visualizations  

$ss_section

## 📦 Tech Stack  
- Python  
- Streamlit  
- TensorFlow  
- Pandas / NumPy  
- Matplotlib & Plotly  

## 📝 License  
MIT License  
"@

# -----------------------------
# 6. Write README.md
# -----------------------------
Set-Content -Path "README.md" -Value $readme -Force
Write-Host "✔ README updated."

# -----------------------------
# 7. Git sync fix (handles remote changes)
# -----------------------------
Write-Host "`n✔ Pulling from GitHub to avoid push rejection..."
git pull origin main --no-edit

# -----------------------------
# 8. Auto commit + push
# -----------------------------
git add .
git commit -m "Auto-updated README, moved screenshots, organized assets"
git push origin main

Write-Host "`n=== Done! Repo updated successfully. ==="
