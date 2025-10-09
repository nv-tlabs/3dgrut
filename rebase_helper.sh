#!/bin/bash

# This script is used to rebase the current branch onto the target branch. The target
# branch is the branch you want to rebase onto. The script will search for the latest
# commit that is the same as the target branch and use it as the base commit.
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
# ./rebase_helper.sh <branch-B>
#
# Commit 95f677a would be identified as the base commit.

TARGET=$1

# Check if the target branch and base commit are provided
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target-branch>"
    exit 1
fi

# Ignore certain files and directories due to git-lfs limitations
ignore=':(exclude)assets/*.gif :(exclude).gitattributes'

# Iterate over commit history backward till the initial commit
commit=$(git rev-parse HEAD)

while [ -n "$commit" ]; do
  echo "Processing commit: $commit"

  # Check if the diff between the commit and the target branch is empty
  if [ "$(git diff $commit $TARGET -- $ignore)" == "" ]; then
    # Print the commit message in one line
    echo $(git log -1 --oneline $commit)
    break
  fi

  # Get the parent commit, if it exists
  parent_commit=$(git rev-parse "$commit^" 2>/dev/null)

  # If there is no parent commit (i.e., we're at the initial commit), break the loop
  if [ $? -ne 0 ]; then
    break
  fi

  # Move to the parent commit
  commit=$parent_commit
done

echo "Commits to be Rebased:"
git log --oneline ${commit}..HEAD

echo
echo "Start Rebasing"
git rebase --onto ${TARGET} ${commit}
