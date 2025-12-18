after fetching from dms,
do we store data to both calls and customerdb? and wt about service_records

curl -X GET "https://api.fortellis.io/cdk/drive/department" \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjlhNTU2ZGYwLWI3NjUtMTFlOC1hMmU4LTI5ZjRlYmZhMjg4MiJ9.eyJjaWQiOiJOVzdEM3pZTmhzU3JkZ01CcGRaS2pBV1ZYdU5Ob0xjYyIsImp0aSI6IkFULkt5NTJCWTZfTWdrc0VIZ0x2Mm0zZ05qQnZsdjYtblF2Y3lyaENKak80cEUiLCJzY3AiOlsiYW5vbnltb3VzIl0sInN1YiI6Ik5XN0QzellOaHNTcmRnTUJwZFpLakFXVlh1Tk5vTGNjIiwiaWF0IjoxNzY1OTgwNDE1LCJleHAiOjE3NjU5ODQwMTUsImF1ZCI6ImFwaV9wcm92aWRlcnMiLCJpc3MiOiJodHRwczovL2lkZW50aXR5LmZvcnRlbGxpcy5pby9vYXV0aDIvYXVzMXAxaXh5N1lMOGNNcTAycDcifQ.buwbj5NwRtt2k_ol2ba9I110r6ebr7-RSg6jRU-q2pgsaNunK300kgn3iv0To5fjLSPP1fCCIXskl83fnPtS4ccCJ9x9N6xbUjkfkJ11a17PMQsTTRgMAAF5kFybIYmQ1WnFEDbV1ENvbyICzF0zWWHs_HmpZKxkvhbRMsWXleiJYcL5Z3nlqi_yVRrzLTWun1czxFbtuRMlwrL2UiuhOf4wgbgo_WRgd9KCL6HWgfQ_FIuouD2gQiZ-N2ttSsEczatfQVex4NW7YOZ9TPYF1P4MeAgV6TezM3m4dsT6OjdPXgRMmXcqhZnzRhzJeXCKQrAveSLa0L65D4jcCjlI0g" \
  -H "Subscription-Id: 55b271af-4e21-4c95-a6d1-5411bbd6c7bd" \
  -H "Request-Id: $(uuidgen)" \
  -H "Accept: application/json"
D100152200
![[Pasted image 20251218181919.png]]
{
"data": [
{
"roNumber": "284399",
"hostItemId": "NH333333*33333",
"addOnFlag": false,
"apptDate": "2022-12-10",
"apptTime": "11:30:00",
"apptFlag": true,
"blockAutoMsg": true,
"bookedDate": "2022-12-20",
"bookedTime": "08:29:52",
"bookerNo": 11111,
"cashier": "gvalker",
"closedDate": "2022-12-20",
"comebackFlag": false,
"comments": "ENGINE IS INSTALLED. HYBRID BATTERY IS DEAD.",
"contactEmailAddress": "ABCD@GMAIL.COM",
"contactPhoneNumber": "23232323232",
"customer": {
"customerId": "111111",
"name1": "NICKLIN,ALLISON J",
"name2": null,
"address": {
"addressLine1": "11 ABCDd CT",
"addressLine2": "xyz US",
"cityStateZip": "FORT WAYNE IN 11111"
},
"phoneNumbers": [
{
"number": "1111111111",
"extension": null,
"description": "HomePhone"
},
{
"number": "2222222222",
"extension": null,
"description": "WorkPhone"
},
{
"number": "3333333333",
"extension": null,
"description": "Cellular"
},
{
"number": "4444444444",
"extension": null,
"description": "MainPhone"
}
],
"emailAddresses": [
{
"address": "abcd@yahoo.com",
"desc": "HOME"
},
{
"address": "abcd@company.com",
"desc": "WORK"
}
]
},
"dedMultiValueCount": 2,
"deductibles": [
{
"sequenceNo": 2,
"lineCodes": "A",
"laborType": "IHPS",
"maximumAmount": 525.01,
"actualAmount": 500.01,
"laborAmount": 468.52,
"partsAmount": 31.48
}
],
"discounts": [
{
"appliedBy": "C",
"classOrType": "C",
"debitAccountNo": "",
"debitControlNo": 1,
"debitTargetCo": "",
"desc": "$10.00 OFF ANY SERVICE PARTS",
"id": "123",
"level": "LOP",
"lineCode": "A",
"lopSeqNo": 1,
"managerOverride": false,
"originalDiscount": 3.25,
"overrideAmount": 3.10,
"overrideGPAmount": 0.01,
"overrideGPPercent": 0.0125,
"overridePercent": 0.0110,
"overrideTarget": 0.01,
"sequenceNo": 5,
"partsDiscount": 0.01,
"laborDiscount": -10.01,
"totalDiscount": -10.01,
"userId": "docxample"
}
],
"disMultiValueCount": 2,
"emailMultiValueCount": 2,
"estimatedCompletionDate": "2022-12-10",
"estimatedCompletionTime": "12:42:00",
"feeMultiValueCount": 2,
"hasCustPayFlag": false,
"hasIntPayFlag": false,
"hasWarrPayFlag": false,
"hrsMultiValueCount": 2,
"lastServiceDate": "2022-08-24",
"warMultiValueCount": 2,
"lbrMultiValueCount": 2,
"linMultiValueCount": 2,
"operations": [
{
"line": {
"actualWork": "2",
"lineCode": "A",
"complaintCode": "H7500",
"serviceRequest": "PERFORM COURTESY MULTI-POINT INSPECTION",
"campaignCode": "",
"addOnFlag": false,
"statusCode": "",
"statusDesc": "",
"comebackFlag": false,
"dispatchCode": "",
"estimatedDuration": "0.00",
"cause": "",
"storySequenceNo": 1,
"storyEmployeeNo": "11111",
"storyText": "5912 PERFORM MULTI POINT VEHICLE INSPECTION",
"bookerNo": 111111,
"laborOperations": [
{
"sequenceNo": 1,
"lineCode": "A",
"opCode": "PFC",
"opCodeDesc": "",
"type": "CPPM",
"mcdPercentage": 100.85,
"forcedShopCharge": 1.25,
"sale": 18.36,
"cost": 4.8,
"technicianIds": [
"99"
],
"bookedDate": "2022-12-20",
"bookedTime": "08:29:52",
"flagHours": "0.00",
"actualHours": "4.00",
"soldHours": "0.3",
"otherHours": "0.00",
"timeCardHours": "0.00",
"comebackFlag": false,
"comebackRO": "",
"comebackSA": 123,
"comebackTech": 456,
"parts": [
{
"sequenceNo": 1,
"lineCode": "A",
"laborSequenceNo": 1,
"number": "11111-2S000",
"bin1": "1102",
"partClass": "",
"comp": "0.00",
"compLineCode": "",
"coreCost": "0.00",
"coreSale": "0.00",
"cost": "10.56",
"desc": "SERVICE KIT-OIL FILTER",
"employeeId": 111111,
"extendedCost": "10.56",
"extendedSale": "18.61",
"laborType": "CPPM",
"list": "17.03",
"mcdPercentage": "110.25",
"outsideSalesmanId": 123,
"qtyBackOrdered": 0,
"qtyFilled": 1,
"qtyOnHand": 44,
"qtyOrdered": null,
"qtySold": 12.25,
"sale": "18.61",
"source": "111",
"specialStatus": "",
"unitServiceCharge": "0.00",
"fees": [
{
"cost": "0.00",
"id": "FRT",
"laborType": "CH",
"lineCode": "E",
"lopOrPartFlag": "P",
"lopOrPartSeqNo": 1,
"mcdPercentage": 8.25,
"opCode": "9997",
"opCodeDesc": "FREIGHT",
"sale": "120.00",
"sequenceNo": 1,
"type": "M"
}
]
}
],
"fees": [
{
"cost": "0.00",
"id": "FRT",
"laborType": "CH",
"lineCode": "E",
"lopOrPartFlag": "P",
"lopOrPartSeqNo": 1,
"mcdPercentage": 5.10,
"opCode": "9997",
"opCodeDesc": "FREIGHT",
"sale": "120.00",
"sequenceNo": 1,
"type": "M"
}
],
"hours": [
{
"sequenceNo": 1,
"lineCode": "A",
"laborType": "CB",
"mcdPercentage": 5.25,
"technicianId": 4038,
"hourType": "ST",
"percentage": "100.00",
"sale": "102.00",
"cost": "16.20",
"actualHours": "0",
"flagHours": "0",
"soldHours": "0.6",
"otherHours": "0",
"timeCardHours": "0"
}
]
}
],
"warrantyList": [
{
"lineCode": "A",
"laborSequenceNo": "1",
"failureCode": "",
"failedPartNo": "865",
"failedPartsCount": 1,
"claimType": "CPN",
"authorizationCode": "",
"conditionCode": ""
}
]
}
}
],
"mileage": 5912,
"mileageOut": 5913,
"mileageLastVisit": 5913,
"mlsMultiValueCount": 2,
"mls": [
{
"sequenceNo": "6",
"lineCode": "D",
"type": "S",
"opCode": "9999",
"opCodeDesc": "RENTAL",
"laborType": "CEXT",
"mcdPercentage": 5.20,
"sale": "0.00",
"gogPrice": "0.00",
"miscPrice": "0.00",
"subletPrice": "280.00",
"gogCost": "280.00",
"subletCost": "0.00",
"miscCost": "0.00",
"cost": "0.00"
}
],
"openDate": "2022-12-10",
"openTime": "11:41:37",
"origPromisedDate": "2022-12-19",
"origPromisedTime": "18:00:00",
"isOrigWaiter": false,
"payCPTotal": "55.00",

"payMultiValueCount": 1,
"totalPaymentMade": "0",
"payBalanceDue": "81.92",
"payments": [
{
"insuranceFlag": false,
"paymentAmount": "0.00",
"code": "CASH"
}
],
"phoneMultiValueCount": 0,
"postedDate": "2022-12-14",
"priorityValue": 1234,
"promisedDate": "2022-12-19",
"promisedTime": "16:12:00",
"prtMultiValueCount": 5,
"punMultiValueCount": 1,
"purchaseOrderNo": "18612",
"rapApptId": 2006651783784,
"rapMultiValueCount": 1,
"remarks": "",
"rentalFlag": false,
"serviceAdvisor": 99,
"isSoldByDealer": false,
"isSpecialCustomer": false,
"statusCode": "C98",
"statusDesc": "CLOSED",
"tagNo": "T1503",
"technicianPunchTimes": [
{
"technicianId": 113067,
"workDate": "2022-12-28",
"timeOn": "12:48:00",
"timeOff": "13:17:00",
"duration": "0.48",
"lineCode": "A",
"workType": "W",
"alteredFlag": false
}
],
"totMultiValueCount": 3,
"totals": [
{
"payType": "C",
"roSale": "1189.07",
"roCost": "340.66",
"laborSale": "843.15",
"laborCost": "187.20",
"laborCount": 4,
"laborDiscount": "0.00",
"laborSalePostDed": "843.15",
"partsSale": "288.09",
"partsCost": "153.46",
"partsCount": 5,
"partsDiscount": "0.05",
"partsSalePostDed": "288.09",
"miscSale": "0.00",
"miscCost": "0.00",
"miscCount": 1,
"lubeSale": "0.00",
"lubeCost": "0.00",
"lubeCount": 1,
"subletSale": "0.00",
"subletCost": "0.00",
"subletCount": 1,
"shopChargeSale": "0.00",
"shopChargeCost": "0.00",
"roTax": "0.00",
"stateTax": "0.00",
"localTax": "0.00",
"supp2Tax": "0.00",
"supp3Tax": "0.00",
"supp4Tax": "0.00",
"actualHours": "0",
"soldHours": "0",
"otherHours": "0",
"timeCardHours": "0.00",
"coreCost": "0.00",
"coreSale": "0.00",
"discount": "0.00",
"flagHours": "0.00",
"forcedShopCharge": "0.00",
"roSalePostDed": "1189.07"
},
{
"payType": "W",
"roSale": "1189.07",
"roCost": "340.66",
"laborSale": "843.15",
"laborCost": "187.20",
"laborCount": 4,
"laborDiscount": "0.00",
"laborSalePostDed": "843.15",
"partsSale": "288.09",
"partsCost": "153.46",
"partsCount": 5,
"partsDiscount": "0.05",
"partsSalePostDed": "288.09",
"miscSale": "0.00",
"miscCost": "0.00",
"miscCount": 1,
"lubeSale": "0.00",
"lubeCost": "0.00",
"lubeCount": 1,
"subletSale": "0.00",
"subletCost": "0.00",
"subletCount": 1,
"shopChargeSale": "0.00",
"shopChargeCost": "0.00",
"roTax": "0.00",
"stateTax": "0.00",
"localTax": "0.00",
"supp2Tax": "0.00",
"supp3Tax": "0.00",
"supp4Tax": "0.00",
"actualHours": "0",
"soldHours": "0",
"otherHours": "0",
"timeCardHours": "0.00",
"coreCost": "0.00",
"coreSale": "0.00",
"discount": "0.00",
"flagHours": "0.00",
"forcedShopCharge": "0.00",
"roSalePostDed": "1189.07"
},
{
"payType": "I",
"roSale": "1189.07",
"roCost": "340.66",
"laborSale": "843.15",
"laborCost": "187.20",
"laborCount": 4,
"laborDiscount": "0.00",
"laborSalePostDed": "843.15",
"partsSale": "288.09",
"partsCost": "153.46",
"partsCount": 5,
"partsDiscount": "0.05",
"partsSalePostDed": "288.09",
"miscSale": "0.00",
"miscCost": "0.00",
"miscCount": 1,
"lubeSale": "0.00",
"lubeCost": "0.00",
"lubeCount": 1,
"subletSale": "0.00",
"subletCost": "0.00",
"subletCount": 1,
"shopChargeSale": "0.00",
"shopChargeCost": "0.00",
"roTax": "0.00",
"stateTax": "0.00",
"localTax": "0.00",
"supp2Tax": "0.00",
"supp3Tax": "0.00",
"supp4Tax": "0.00",
"actualHours": "0",
"soldHours": "0",
"otherHours": "0",
"timeCardHours": "0.00",
"coreCost": "0.00",
"coreSale": "0.00",
"discount": "0.00",
"flagHours": "0.00",
"forcedShopCharge": "0.00",
"roSalePostDed": "1189.07"
}
],
"vehicle": {
"vin": "ABCD1234PQR",
"deliveryDate": "2020-02-06",
"make": "GENE",
"makeDesc": "GENESIS",
"model": "G70",
"modelDesc": "G70",
"year": 2020,
"licenseNumber": "111xxx",
"vehicleColor": "WHITE",
"vehId": "AAA1111",
"lotLocation": ""
},
"visItemMultiValueCount": 2,
"voidedDate": "",
"isCustomerWaiting": false
}
]
}