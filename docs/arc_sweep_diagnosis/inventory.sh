#!/usr/bin/env bash
# One line per simulation run: what it was configured as, and what it returned.
cd /home/pk/.claude/jobs/10fee75c/tmp || exit 1
printf "%-16s %-8s %-6s %-46s %s\n" run arc envs intervention result
for f in fix_*.log; do
  t=${f#fix_}; t=${t%.log}
  arc=$(grep -a "^\[cfg\]" "$f" 2>/dev/null | tail -1 | sed -n 's/.*arc_std=\([^ ]*\).*/\1/p')
  envs=$(grep -a "num_envs" "$f" 2>/dev/null | head -1 | sed -n 's/.*num_envs[= ]\([0-9]*\).*/\1/p')
  fx=$(grep -a "^\[fix\]" "$f" 2>/dev/null | sed 's/\[fix\] //' | tr '\n' ';' | cut -c1-46)
  res=$(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$f" 2>/dev/null | tail -1 | sed 's/ successful//')
  printf "%-16s %-8s %-6s %-46s %s\n" "$t" "${arc:-?}" "${envs:-?}" "${fx:--}" "${res:-INCOMPLETE}"
done
