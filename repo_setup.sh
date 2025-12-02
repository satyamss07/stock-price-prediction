#!/bin/bash

echo "=== Organizing repo ==="

# 1. Create assets folder
mkdir -p assets

# 2. Move screenshots
mv *.png assets/ 2>/dev/null
mv *.jpg assets/ 2>/dev/null
mv *.jpeg assets/ 2>/dev/null

echo "✔ All images moved into assets/"

# 3. Auto-generate screenshots section
ss_section="## Screenshots\n\n"
for f in assets/*; do
  ss_section+="![Screenshot]($f)\n\n"
done

# 4. Create README
cat > README.md <<EOF
# Stock Price Prediction (LSTM + Streamlit)

A complete stock prediction + analysis app with LSTM, real-time data, technical indicators & recommendation engine.

---

## Features
- Real-time stock data (Yahoo Finance)
- Custom date range (up to 10 years)
- Technical indicators: SMA, RSI, Volatility
- LSTM deep learning prediction model
- Buy / Hold / Avoid recommendation
- Download predictions as CSV

$ss_section

## Tech Stack
- Python
- Streamlit
- TensorFlow
- Pandas / NumPy
- Plotly / Matplotlib

## License
MIT License
EOF

echo "✔ README.md updated"

# 5. Git sync + push
git pull origin main --no-edit
git add .
git commit -m "Auto update: README + assets"
git push origin main

echo "=== Done! Repo updated successfully ==="
