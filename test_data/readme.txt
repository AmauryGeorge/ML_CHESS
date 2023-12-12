The following columns are available in the mydump.csv
Every row represents a move of a game.

id: is the id of the selected move
gameId: is the id of the corresponding game
turnNumber: in which turn the move was done (per turn there are two moves)
number: counts how many moves are done in the game
move_state: if a move was manually confirmed("valid") or if it was automatically confirmed("autovalid")
confidence: predicted confidence for the predicted move
gl: google API output
gl2: second google API output
az: azure API output
rk: amazon API output
ab: abby API output
prediction: For autovalid automatically confirmed move and for valid from the user entered move
width: image width
height: image height
mimetype: image type


The images can be found in the folder "images"
For every row, there should be a corresponding image.
If we take the first row, we look at the image id for example 131, the image names are called as follows: "id.*" for the first row it would be 131.png

* are the different image types