#!/usr/bin/env bash
# Creates the study yaml tags study-S0 / S1 / S2 / ours / TK / TKC WITHOUT touching main:
#   S0   = the current yaml on main (HEAD)
#   S1/S2/ours/TK/TKC = HEAD + study/<S>.patch committed on a detached HEAD, tagged, then back to the branch.
# Run only after the NS2 and Pure-INDI decisions. Default is a dry run; pass --execute to create the tags (local), then `git push --tags`.
#   crazyflie/config/study/make_tags.sh [--execute]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXEC=0; [ "${1:-}" = "--execute" ] && EXEC=1
BR=$(git rev-parse --abbrev-ref HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "working tree has uncommitted changes to tracked files: commit or stash first"; exit 1; }
for t in S0 S1 S2 ours TK TKC; do git rev-parse -q --verify "refs/tags/study-$t" >/dev/null && { echo "tag study-$t already exists"; exit 1; }; done
echo "branch $BR, HEAD $(git rev-parse --short HEAD)"
echo "study-S0   -> HEAD (current yaml)"
for t in S1 S2 ours TK TKC; do git apply --check "crazyflie/config/study/$t.patch" && echo "study-$t -> HEAD + $t.patch (applies cleanly)"; done
[ $EXEC -eq 1 ] || { echo "dry run: nothing created (use --execute)"; exit 0; }
git tag study-S0
git checkout -q --detach HEAD
for t in S1 S2 ours TK TKC; do
  git checkout -q -- crazyflie/config/crazyflies.yaml          # always from the S0 yaml
  git apply "crazyflie/config/study/$t.patch"
  git commit -q -m "study yaml $t (frozen block of docs/comparative_study_protocol.md section 2)" -- crazyflie/config/crazyflies.yaml
  git tag "study-$t"
  git reset -q --hard "study-S0"
done
git checkout -q "$BR"
git tag -l 'study-*'
echo "created locally; push with: git push origin --tags"
