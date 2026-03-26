#!/bin/bash
# Hook handler for Claude Code — captures tool use events for debugging.
# Stdin receives JSON event data from Claude Code hooks.

input=$(cat)
output_file="/tmp/hook.out"
echo -e "${input}\n\n" >> "$output_file"
exit 0
