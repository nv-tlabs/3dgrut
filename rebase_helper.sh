#!/bin/bash

# This script is used to rebase the current branch onto the target branch. The base 
# commit is the commit prior to the earliest commits you want to rebase. The target
# branch is the branch you want to rebase onto.
#
#  Branch A                    Branch B <target-branch>
#  | 2bce381                   | a7da927 
#  | 895eb6d                   | e069629
#  | 1e9f10f                   | ...
#  | f3db518                   | ...
#  | c6be096                   | ...
#  | f50f78b                   | ...
#  | 95f677a <base-commit>     | ...
#  | ...                       | ...
#
# For example, if you want to rebase commits from f50f78b to 2bce381 from branch A to branch B,
# you can run the following command:
#
# ./rebase_helper.sh <branch-B> 95f677a
#

TARGET=$1
BASE=$2

# Check if the target branch and base commit are provided
if [ -z "$TARGET" ] || [ -z "$BASE" ]; then
    echo "Usage: $0 <target-branch> <base-commit>"
    exit 1
fi

echo "Commits to be Rebased:"
git log --oneline ${BASE}..HEAD

echo 
echo "Start Rebasing"
git rebase --onto ${TARGET} ${BASE}
