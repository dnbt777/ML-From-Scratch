# Backpropagation

Calculate the direction the error goes in, then go the opposite direction

Do this by determining how sensitive the error is to each component you want to change

Changes stack up. If a changes b changes c, you can calculate the effect of a on c

![image](https://github.com/dnbt777/ML-From-Scratch/assets/169108635/90bf6ee2-91cb-43a3-8503-14bf503c8b71)

Go down through the chain, the dz/da*da/dz is the 'delta' to keep track of. At every layer there is a branch, and at each branch calculate the layer's dl/dw and dl/db, then add it to the updates



## Speedrunning

Records
 - 1.4hrs (5/14)
 - 20hrs (5/14)