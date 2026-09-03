#!/usr/bin/env bash
# Pretend to be a fresh session with only the documents: can each handoff question be answered?
# A question that lands in zero files is a gap; one that lands in several is fine as long as they
# agree, which is why the routing index exists.
cd /home/pk/IsaacLab/.claude/worktrees/mimic-bugfix || exit 1
DOCS="CLAUDE.md docs/主线二_思维链交接.md docs/README_主线二.md docs/接触锚定扰动增广_设计记录.md"
DOCS="$DOCS docs/arc_sweep_diagnosis/README.md docs/arc_sweep_diagnosis/REMEDIES.md docs/arc_sweep_diagnosis/EXPERIMENT_LEDGER.md"

check () {
  n=$(grep -l -- "$2" $DOCS 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then printf "  MISSING  %s\n" "$1"; else printf "  ok (%s) %s\n" "$n" "$1"; fi
}

echo "handoff self-check"
check "current headline conclusion (no harm, no benefit)"        "既无害也无益"
check "why arc costs generation success"                          "冻结的是目标"
check "MimicGen replays targets, not achieved poses"              "target_eef_pose"
check "which runs may be compared"                                "EXPERIMENT_LEDGER"
check "which runs are invalid"                                    "INVALID"
check "the five silent-no-op failures"                            "静默失效"
check "direction 2 result and why it is not a fair test"          "仍未被公平测试"
check "the 0.69 cm evidence"                                      "0.69"
check "next-step options"                                         "下一步的分叉"
check "retracted claims are flagged"                              "已作废"
check "PR numbers and status"                                     "#7434"
check "where the scripts and logs live"                           "arc_sweep_diagnosis"
check "success criteria, two tiers"                               "第二层"
check "how to run develop tests locally"                          "PYTHONPATH"
