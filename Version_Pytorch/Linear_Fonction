# ===================================================================
# Corentin : Voici une version plus simple que le NeuralNet from scratch,
# il ne lit que les fonctions linéaires, je m'en sers en entrainement sur PyTorch
# ===================================================================


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Gen data
d = torch.linspace(0, 10, 1000).unsqueeze(1)
print("data:\n", d)

#Gen bruit
sigma = 0.1
r = torch.randn(1000,1) * sigma
print("Bruit:\n", r)

#Fonction réelle
y=256*d+39+r
print("y:\n", y)

#création modèle
model= nn.Linear(1, 1)

#calcul la moyenne des erreurs au carré
criterion = nn.MSELoss()
#met à jour les poids en fonction du gradient
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#boucle
for epoch in range(800):
    y_pred = model(d)    #prédire
    loss = criterion(y_pred, y)    #calculer la perte
    optimizer.zero_grad()   #mettre à zéro les gradients
    loss.backward() #calculer les gradients
    optimizer.step()    #mettre à jour les poids
    if epoch % 20 == 0:
        print(f'[{epoch:03d}] loss: {loss.item():.6f}')



print(list(model.parameters()))

with torch.no_grad():                  # désactive le calcul des gradients
    y_pred = model(d)

w = model.weight.item()
b = model.bias.item()
print(f"Fonction estimée : y = {w:.3f}x + {b:.3f}")


plt.scatter(d, y)
plt.plot(d, y_pred.detach().numpy(), color='red')  #tracer la
plt.xlabel('d')
plt.ylabel('y')
plt.show()

