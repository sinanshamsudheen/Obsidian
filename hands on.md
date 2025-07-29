saturday 6:30
wts for next week, next month
where are we going,
wts happening

the engineers call

ps -eo pid,cmd,rss | grep "/home/zero/Packages/zen/zen" | grep -v grep | awk '{sum += $3} END {print "Zen RAM Usage: " sum/1024 " MB"}'