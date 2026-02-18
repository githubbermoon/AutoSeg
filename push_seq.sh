#!/bin/bash
# script to push changes one by one in segT

echo "Checking for changes..."
files=$(git ls-files -m)
untracked=$(git ls-files -o --exclude-standard)

all_files="$files
$untracked"

# Filter out empty lines
all_files=$(echo "$all_files" | grep -v '^$')

if [ -z "$all_files" ]; then
    echo "No changes found."
    exit 0
fi

echo "Found changes in:"
echo "$all_files"
echo "-----------------------------------"

IFS=$'\n'
for file in $all_files; do
    echo ""
    echo "File: $file"
    git diff "$file" 2>/dev/null || cat "$file" # show diff or content if untracked
    echo ""
    read -p "Do you want to commit and push '$file'? (y/n): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        git add "$file"
        read -p "Enter commit message: " msg
        git commit -m "$msg"
        echo "Pushing..."
        git push
        echo "Done with $file."
    else
        echo "Skipping $file."
    fi
done
