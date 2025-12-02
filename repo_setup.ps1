Write-Host "`n=== Organizing repo ==="

# 1. Create assets folder
if (!(Test-Path "assets")) {
    New-Item -ItemType Directory -Force -Path "assets" | Out-Null
}

# 2. Move & rename images
$count = 1
Get-ChildItem -File *.png, *.jpg, *.jpeg | ForEach-Object {
    $ext = $_.Extension
    $newName = "screenshot_$count$ext"
    Move-Item $_.FullName -Destination ("assets/" + $newName) -Force
    $count++
}

Write-Host "✔ Images moved."

# 3. Build README content in small pieces
$header = "# Stock Price Prediction (LSTM + Streamlit)`n`n"
$desc   = "A real-time stock analysis dashboard with LSTM predictions, indicators, and recommendations.`n`n"
$screenHeader = "## Screenshots`n`n"

# Generate screenshot list
$screens = ""
Get-ChildItem assets | ForEach-Object {
    $screens += "### $($_.Name)`n"
    $screens += "<img src='assets/$($_.Name)' width='700'>`n`n"
}

$run = "## Run App`n`nstreamlit run stock2.py`n`n"
$license = "## License`nMIT License"

# 4. Combine everything
$full = $header + $desc + $screenHeader + $screens + $run + $license

# 5. Write README
Set-Content -Path "README.md" -Value $full

Write-Host "✔ README updated."

# 6. Git commit + push
git add .
git commit -m "Auto: organized assets + updated README"
git push

Write-Host "`n=== Done! ===`n"
