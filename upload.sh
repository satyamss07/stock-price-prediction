#!/bin/bash

echo "🚀 Starting GitHub project setup for Stock Price Prediction..."

# Initialize git
git init

# Create main branch
git checkout -b main

# Add all files
git add .

# Commit
git commit -m "Initial commit – Stock Price Prediction App"

# Add remote GitHub repo
git remote add origin https://github.com/satyamss07/stock-price-prediction.git

# Push
git push -u origin main

echo "🎉 Upload complete!"
echo "✔ Your repository is live at:"
echo "👉 https://github.com/satyamss07/stock-price-prediction"
