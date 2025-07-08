#!/bin/bash

TARGET=$1

# Check if the target branch is provided
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target-branch>"
    exit 1
fi

BASE=43947a728a7d171984ee810ebde653288562ae9e

echo "Commits to be Rebased:"
git log --oneline ${BASE}..HEAD

echo 
echo "Start Rebasing"
git rebase --onto ${TARGET} ${BASE}
