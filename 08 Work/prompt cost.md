new prompt(5 few shots),
input: 3500-4000 tokens
output: 100-150 tokens

- **Input:** ~$0.00875–$0.01000
    
- **Output:** ~$0.00150
    
- **Total:** **~$0.01025–$0.01150** per request

select id,name, plan_id, credit_balance from organizations;
![[Pasted image 20250809090710.png]]

vapi returns cost per call through the webhook and that cost is saved inside the cost column of calls table.
![[Pasted image 20250809092201.png]]

![[Pasted image 20250809092819.png]]

testing,
"endedReason": "assistant-ended-call",
"endedReason": "customer-ended-call",
"endedReason": "customer-busy",

busy ac - missed call
declined - try again - missed
pick & cut - completed
if cut -> RETRY LOGIC
NO RESPONSE -> BUSY (MISSED TAG) -> RETRY 

CREATE MAPPINGS,
CURRENT STATUS
COMPLETED, FAILED, MISSED, BUSY, RETRY
RETRY CALLS -> READY
OPT OUT, 

COMPLETED ALSO HAS FAILED



bugs,
1- call pickeythu, completed 
2- call missed, 
3- call failed, 

real life,
opt out(cust said no) _> status? > completed (opted out)

opt out after nps score and feedback(not imp)
google reivew opted out(not imp)






