1- check credit, if no creds, user cant call,
2- generate cost of openai,
result = (use prompt's token size value + openai response token and generate cost(different for both input and output tokens, take that into consideration)) + (generate vapi's cost per min duration) 
3-result += 20% of result
4- deduct result from cost column and update

- Webhooks can retry; without an idempotent guard, you risk double-deducting. Given the current code sets call.cost to VAPI’s number before any combined-cost billing, a naive “only bill when call.cost IS NULL” would never run on first delivery.