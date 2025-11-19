table
 | id (PK) | organization_id (FK)|rooftop_id(FK)| insights|frequency|generated_at | created_at | updated_at



(frontend) areas to improve -> view more insights -> new page -> 2 column page which has all the valid tags -> for each tag we show the top 3 concerns and the frequency of each( top right side) 

Backend,
openai,
input = feedback summary for the date range for that rooftop.



prompt=
our goal is to implement a new feature, 
which will be run every 3 week to generate top 3 insightful summaries for each tag.

we need a openai function which will generate the below output for the input we give.

input = feedback summary for the date range for that rooftop.

a
output = 
{

"service quality": [

{

"concern": "Customer found the brake oil to be of bad quality",

"frequency": 10

},

{

"concern": "Customer found the engine tuning to be of bad quality",

"frequency": 22

},

{

"concern": "Customer found the fan service to be of bad quality",

"frequency": 34

}

],

"staff professionalism": [

{

"concern": "Staff displayed poor communication or provided unclear explanations",

"frequency": 15

},

{

"concern": "Staff were frequently late or did not adhere to scheduled appointment times",

"frequency": 8

},

{

"concern": "Staff lacked technical knowledge or were unable to answer questions about procedures",

"frequency": 12

}

]

}
(same for all the tags invovled(here we only have 2 tags))

here frequency should tell how many times those concerns were raised or how many customers pointed out that concern.
and those concern should also be generated dynamically based on the feedbacks

and then we extract it into the models/insights table so that we can later fetch.

more instructions,
u gotta create or all this setup inside a new folder microservices/ within the parent directory. so that i can deploy it easily for scheduling in the future. so do not write the function within the existing openai_service.py, but u can follow the structure for code but create a new service inside the microservices/ folder.

the instructions given to me by my CTO = "Create the logic and stuff as a part of this story. Write it seperately as a new module in the code. Create a directory called microservices. Then inside that create the lgock. When you run a main file it should do this.  
After you are done we verify this, we can have a new story for the deployment, I can work with you on that."

and dont worry about frontend. another agent is working on the frontend parallely.


