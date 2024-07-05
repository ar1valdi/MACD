$conflictedFiles = git diff --name-only --diff-filter=U

echo $conflictedFiles

$conflictedFiles | ForEach-Object { git add $_ }

git commit -m "Resolved both added conflicts"
