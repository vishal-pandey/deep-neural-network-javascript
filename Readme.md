## Deep Neral Network
---

al  = Activation of last layer  
al1 = Activation of second last layer
bl = bias of last layer
wl  = Weight of last layer
y   = output
C = Cost function

C = (al - y)^2

zl = wl * al1 + bl
al = σ(wl * al1 + bl) = σ(zl)


∂C_∂wl = ∂zl_∂wl * ∂al_∂zl * ∂C_∂al
∂C_∂bl = ∂zl_∂bl * ∂al_∂zl * ∂C_∂al

∂zl_∂wl = al1
∂al_∂zl = σ`(zl)
∂C_∂al = 2 * (al - y)
∂zl_∂bl = 1

∂C_∂wl = al1 * σ'(zl) * 2 * (al - y)
∂C_∂bl =       σ'(zl) * 2 * (al - y)






Similarly
zl1 = wl1 * al2 + bl1
al1 = σ(wl1 * al2 + bl1) = σ(zl)