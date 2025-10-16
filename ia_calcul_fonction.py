import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def make_dataset(n=512, x_range=(-3.0, 3.0), noise=0.05, f=None):
    X = np.random.uniform(x_range[0], x_range[1], size=(n, 1))
    y_true = f(X)
    y_noisy = y_true + np.random.normal(0, noise, y_true.shape)

    return X, y_noisy, y_true, f 


def quick_plot_data(X, y_noisy):
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y_noisy, s=10, alpha=0.5)
    plt.title("Données d'entraînement (avec bruit)")
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout(); plt.show()

class Linear:
    def __init__(self, in_features, out_features, seed=None):
        # 1. Initialisation des poids et biais
        if seed is not None:
            np.random.seed(seed)
        limit = np.sqrt(6 / (in_features + out_features))  # borne de Glorot/Xavier
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)

        # 2. Placeholders pour stocker les gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # 3. Cache pour X (utilisé dans backward)
        self.cache_X = None

    def forward(self, X):
        """
        X: shape (n, in_features)
        Return: Y = X @ W + b, shape (n, out_features)
        """
        self.cache_X = X  # on garde pour le backward
        return X @ self.W + self.b

    def backward(self, dY):
        """
        dY: shape (n, out_features)
        Return: dX: shape (n, in_features)
        """
        dX = dY @ self.W.T
        self.dW = self.cache_X.T @ dY
        self.db = np.sum(dY, axis=0)
        return dX
    
class Tanh:
    def __init__(self):
        self.out = None  # on stocke la sortie du forward

    def forward(self, X):
        """
        X : valeurs reçues de la couche linéaire
        Retourne : tanh(X)
        """
        self.out = np.tanh(X)
        return self.out

    def backward(self, dOut):
        """
        dOut : gradient reçu de la couche suivante
        Retourne : dL/dX = dOut * (1 - tanh(X)^2)
        """
        return dOut * (1 - self.out**2)

class MSELoss:
    def __init__(self):
        self.cache = None  # pour stocker y_pred et y_true

    def forward(self, y_pred, y_true):
        """
        Calcule la valeur de la perte moyenne MSE.
        """
        self.cache = (y_pred, y_true)
        return ((y_pred - y_true) ** 2).mean()

    def backward(self):
        """
        Calcule la dérivée de la perte par rapport à y_pred.
        """
        y_pred, y_true = self.cache
        n = y_true.shape[0]
        return (2.0 / n) * (y_pred - y_true)

class SGD:
    def __init__(self, model, lr=1e-2):
        """
        model : le modèle complet (doit avoir .parameters())
        lr : taux d'apprentissage
        """
        self.model = model
        self.lr = lr

    def step(self):
        """
        Met à jour les poids à partir des gradients courants
        """
        for p, g in self.model.parameters():
            p -= self.lr * g

    def zero_grad(self):
        """
        Réinitialise les gradients
        """
        for _, g in self.model.parameters():
            g[...] = 0.0


class MLP:
    def __init__(self, in_dim=1, hidden=64, out_dim=1, seed=42):
        np.random.seed(seed)
        self.layers = [
            Linear(in_dim, hidden, seed),
            Tanh(),
            Linear(hidden, hidden, seed+1),
            Tanh(),
            Linear(hidden, out_dim, seed+2)
        ]

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dOut):
        grad = dOut
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append((layer.W, layer.dW))
                params.append((layer.b, layer.db))
        return params

    def zero_grad(self):
        for _, g in self.parameters():
            g[...] = 0.0

def train(epochs=500, lr=0.01, hidden=64, f=None):
    X, y, _, f = make_dataset(n=1024, noise=0.05, f=f)
    model = MLP(in_dim=1, hidden=hidden, out_dim=1)
    loss_fn = MSELoss()
    optim = SGD(model, lr=lr)

    history = []
    for ep in range(1, epochs + 1):
        y_pred = model.forward(X)
        loss = loss_fn.forward(y_pred, y)
        dL = loss_fn.backward()
        model.zero_grad()
        model.backward(dL)
        optim.step()
        history.append(loss)
        if ep % 50 == 0:
            print(f"[{ep:04d}] Loss = {loss:.6f}")

    return model, history, f




def visualize_training(model, history, f):
    plt.figure(figsize=(10,4))
    plt.plot(history)
    plt.title("Évolution de la perte (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.grid()
    plt.show()

    # Affichage de la vraie fonction et de la prédiction
    X_plot = np.linspace(-3, 3, 300).reshape(-1,1)
    y_true = f(X_plot)  # on utilise la vraie fonction passée en argument
    y_pred = model.forward(X_plot)

    plt.figure(figsize=(7,5))
    X_sample, y_sample, _, _ = make_dataset(512, f=f)
    plt.scatter(X_sample, y_sample, s=10, alpha=0.4, label="Données bruitées")
    plt.plot(X_plot, y_true, 'g--', label="Vraie fonction")
    plt.plot(X_plot, y_pred, 'r', label="Prédiction du modèle")
    plt.legend(); plt.grid()
    plt.title("Approximation de f(x) par le réseau")
    plt.show()



if __name__ == "__main__":
    EPOCHS = 800
    LR = 0.01
    HIDDEN = 64

    f = lambda X: np.exp(X)  # Fonction à approximer
    print("Démarrage de l’entraînement...")
    model, history, f = train(epochs=EPOCHS, lr=LR, hidden=HIDDEN, f=f)
    print("Entraînement terminé !")

    visualize_training(model, history, f)