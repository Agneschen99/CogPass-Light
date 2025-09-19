#!/usr/bin/env bash
msg="${1:-chore: update}"
git add .
git commit -m "$msg" || echo "No changes to commit."
git push origin main
