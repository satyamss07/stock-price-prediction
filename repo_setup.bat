@echo off
REM repo_setup.bat - reorganize project, create docs/templates, update README, commit & push
REM Run this from your project root: C:\Users\sammy\Downloads\stockmarket

SET REPO_DIR=%~dp0
cd /d "%REPO_DIR%"

echo.
echo --- Repo setup starting in: %REPO_DIR% ---
echo.

REM 1) Create folders
mkdir assets 2>nul
mkdir notebooks 2>nul
mkdir tests 2>nul
mkdir docs 2>nul
mkdir .github 2>nul
mkdir .github\ISSUE_TEMPLATE 2>nul
mkdir .github\workflows 2>nul

REM 2) Move images (png/jpg/jpeg) to assets
echo Moving image files into assets\ ...
for %%f in ("*.png" "*.jpg" "*.jpeg") do (
  if exist "%%~f" move /Y "%%~f" "assets\" >nul
)

REM 3) Create CONTRIBUTING.md
powershell -Command "Set-Content -Path 'CONTRIBUTING.md' -Value @'
# Contributing

Thanks for your interest in contributing!

Steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-change`.
3. Make changes and add tests where appropriate.
4. Commit your changes: `git commit -m \"Feature: brief description\"`.
5. Push your branch: `git push origin feature/your-change`.
6. Open a Pull Request describing your changes.

Be respectful and follow the existing code style. Thanks!
'@ -Encoding UTF8"

REM 4) Create CODE_OF_CONDUCT.md
powershell -Command "Set-Content -Path 'CODE_OF_CONDUCT.md' -Value @'
# Code of Conduct

We expect contributors to behave respectfully. Please follow the Contributor Covenant principles: be welcoming, respectful, and collaborative.

Report any issues to the repository owners.
'@ -Encoding UTF8"

REM 5) Create ISSUE_TEMPLATE (bug report)
powershell -Command "Set-Content -Path '.github\\ISSUE_TEMPLATE\\bug_report.md' -Value @'
---
name: Bug report
about: Report a bug in the application
---

### Describe the bug
A clear and concise description of the bug.

### To Reproduce
Steps to reproduce:
1. ...
2. ...
3. ...

### Expected behavior
What you expected to happen.

### Environment
- OS:
- Python version:
- Streamlit version:
'@ -Encoding UTF8"

REM 6) Create PULL_REQUEST_TEMPLATE.md
powershell -Command "Set-Content -Path '.github\\PULL_REQUEST_TEMPLATE.md' -Value @'
## Summary

(Brief description of changes.)

### Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update

### How to test
(Explain how reviewers can test your changes.)
'@ -Encoding UTF8"

REM 7) Create DEPLOY.md
powershell -Command "Set-Content -Path 'DEPLOY.md' -Value @'
# Deploy

## Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Connect your GitHub account
3. Select this repository
4. Set run command: `streamlit run stock2.py`
5. Deploy

## Docker (optional)
Build and run:
