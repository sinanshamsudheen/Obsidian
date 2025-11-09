table
 | id (PK)| org_id(FK)|rooftop_id(FK)| insights|frequency|generated_at


areas to improve -> view more insights -> new page -> 2 column page which has all the valid tags -> for each tag we show the top 3 concerns and the frequency of each( top right side) 


openai,
input = feedback summary for the date range for that rooftop.

output = 
{

"service quality": {

"brake oil change": {

"concern": "Customer found the brake oil to be of bad quality",

"frequency": 10

},

"engine tuning": {

"concern": "Customer found the engine tuning to be of bad quality",

"frequency": 22

},

"fan service": {

"concern": "Customer found the fan service to be of bad quality",

"frequency": 34

}

},

"staff professionalism": {

"communication": {

"concern": "Staff displayed poor communication or provided unclear explanations",

"frequency": 15

},

"punctuality": {

"concern": "Staff were frequently late or did not adhere to scheduled appointment times",

"frequency": 8

},

"technical competency": {

"concern": "Staff lacked technical knowledge or were unable to answer questions about procedures",

"frequency": 12

}

}

}

![[Pasted image 20251109222506.png]]


prompt=
our goal is to implement a new feature, 
which will be run every 3 week to generate top 3 insightful summaries for each tag.

we need a openai function which will generate the below output for the input we give.

input = feedback summary for the date range for that rooftop.

a
output = 
{

"service quality": {

"brake oil change": {

"concern": "Customer found the brake oil to be of bad quality",

"frequency": 10

},

"engine tuning": {

"concern": "Customer found the engine tuning to be of bad quality",

"frequency": 22

},

"fan service": {

"concern": "Customer found the fan service to be of bad quality",

"frequency": 34

}

},

"staff professionalism": {

"communication": {

"concern": "Staff displayed poor communication or provided unclear explanations",

"frequency": 15

},

"punctuality": {

"concern": "Staff were frequently late or did not adhere to scheduled appointment times",

"frequency": 8

},

"technical competency": {

"concern": "Staff lacked technical knowledge or were unable to answer questions about procedures",

"frequency": 12

}

}

}
(same for all the tags invovled(here we only have 2 tags))

and then we extract it into a table so that we can later fetch 