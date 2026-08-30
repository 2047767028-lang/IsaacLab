#!/usr/bin/env bash
# The RESULT line never prints -- env_loop closes the env before main() resumes -- so read the last
# progress line each run wrote instead. Same numbers, one step earlier in the pipeline.
cd /home/pk/.claude/jobs/10fee75c/tmp || exit 1
printf "%-12s %-9s %-7s %-7s %s\n" run arc_m dwell scale result
for tag in ref_low ref_high dwell_high scale_high dwell_low scale_low; do
  f="fix_$tag.log"
  if [ ! -f "$f" ]; then printf "%-12s %s\n" "$tag" "(not started)"; continue; fi
  arc=$(grep -a "^\[cfg\]" "$f" | tail -1 | sed -n 's/.*arc_std=\([^ ]*\).*/\1/p')
  dwell=$(grep -a "num_fixed_steps" "$f" | tail -1 | sed -n 's/.*= \([0-9]*\) on.*/\1/p')
  scale=$(grep -a "arm_action.scale" "$f" | tail -1 | sed -n 's/.*-> //p')
  last=$(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$f" | tail -1)
  printf "%-12s %-9s %-7s %-7s %s\n" "$tag" "${arc:-?}" "${dwell:-0}" "${scale:-0.5}" "${last:-running}"
done
