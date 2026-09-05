#!/usr/bin/env bash
# Stop any running teleop session + CloudXR runtime.
pkill -f "teleop_se3_ag""ent" 2>/dev/null
pkill -f "isaacteleop.cloudxr.runt""ime" 2>/dev/null
sleep 2
echo "stopped"
