You Only Look Once
![[Pasted image 20250604133041.png]]![[Pasted image 20250604133143.png]]
this works well for single objects, wt if there are n number of objects?
Yolo will divide the image into grids, and calculate for each grid,

![[Pasted image 20250604133446.png]]
![[Pasted image 20250604133528.png]]
![[Pasted image 20250604133614.png]]
we only have to pass once and no multiples iternation are needed

Issues for this approach,
1- Algo may detect multiple bounding boxes for the same image. 
soln - IOU (intersection over union) = intersect area/union area
![[Pasted image 20250604133842.png]]
discard the rects which had IOU > 0.65 and kept the rect which had class probability as max.

2- What if one grid cell has center of two objects?
![[Pasted image 20250604134122.png]]
soln- Multiple anchor boxes(concatenated)
![[Pasted image 20250604134219.png]]
![[Pasted image 20250604134248.png]]