name, number, { previous_visits(all)-> service_type, nps score, vehicle make/model }, total payment, total_spend_by_customer (excluding warranty pay or internal pay), 

main page UI,
name, number, vehicle, last nps score, year to date spend and a view details button

when user clicks on view details -> { previous_visits(all)-> service_type, nps score, vehicle make/model }total payment, total_spend_by_customer (excluding warranty pay or internal pay),  etc (shown as list)

and we r gonna include a chatbot like agent which can fetch data from queries such as 'find me the customers who spend more than 60k'



do we store the raw data in db or s3
"payments": [

{

"insuranceFlag": false,

"paymentAmount": "658.82",

"code": "CASH"

}

],