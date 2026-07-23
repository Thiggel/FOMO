# -------------------------------
# Abgabegruppe:
# Personen:
# HU-Accountname:
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt


def load_honda_city_dataset():
    """
    Liest den Honda City Datensatz als NumPy Arrays ein.

    Output:

    year_built = np.array[int],
    km_driven = np.array[int],
    selling_price = np.array[float]
    """
    X = np.loadtxt("honda_city.csv", delimiter=",", skiprows=1, dtype=float)

    year_built = np.asarray(X[:,0], dtype=int)
    km_driven = np.asarray(X[:,1], dtype=int)
    selling_price = X[:,2]

    return year_built, km_driven, selling_price


def gradient_descent(X_norm, y, theta, alpha=0.02, epochs=250):
    m = len(y)

    # Add intercept column
    X_b = np.column_stack((np.ones((m, 1)), X_norm))

    losses = []

    for _ in range(epochs):
        error = X_b @ theta - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        gradient = (2 / m) * X_b.T @ error
        theta -= alpha * gradient

    return theta, np.array(losses)


def teilaufgabe_a():
    """
    Implementiert eine multivariate lineare Regression mittels Gradient Descent, um den Verkaufspreis
    von Honda City Fahrzeugen basierend auf dem Baujahr und dem Kilometerstand zu schätzen.

    Die Funktion lädt den Datensatz, erstellt die normalisierte Merkmalsmatrix, initialisiert den
    Parametervektor mit Nullen und führt Gradient Descent zur Optimierung bis zur Konvergenz durch.

    Output:
        theta = np.array[float],
        losses = np.array[float]
    """
    year_built, km_driven, selling_price = load_honda_city_dataset()

    # Featurematrix erstellen
    X = np.column_stack((year_built, km_driven))

    # Features normalisieren, damit Gradient Descent schneller und stabiler konvergiert
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma

    # Parametervektor deterministisch initialisieren
    theta = np.zeros((X.shape[1] + 1, 1))
    # Verkaufspreise als Spaltenvektor darstellen
    y = selling_price.reshape(-1, 1)

    theta, losses = gradient_descent(X_norm, y, theta)

    return theta, losses


def teilaufgabe_b():
    """
    Visualisiert den Verlauf des mittleren quadratischen Fehlers (MSE) über die Iterationen des
    Gradient Descent, wie er in teilaufgabe_a() berechnet wurde.

    Die Funktion ruft teilaufgabe_a() auf, extrahiert die Verlustwerte (Losses) und stellt sie
    als Liniendiagramm mit beschrifteten Achsen dar.

    Output:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots()

    _, losses = teilaufgabe_a()

    ax.plot(losses)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Loss Curve')
    ax.grid(True)
    fig.tight_layout()

    '''
    Bedeutung: Der mittlere quadratische Fehler (MSE) misst die durchschnittliche quadratische Abweichung zwischen den vorhergesagten und den tatsächlichen Verkaufspreisen (in Tausend Dollar). Die Einheit des MSE ist daher (Tausend Dollar)². Ein kleiner MSE-Wert deutet darauf hin, dass das Modell den Verkaufspreis gut vorhersagen kann. Ein großer Wert hingegen bedeutet, dass die Vorhersagen stark vom tatsächlichen Preis abweichen.
    '''

    return fig


if __name__ == "__main__":
    print(f"Teilaufgabe a:\n{teilaufgabe_a()}")
    
    fig = teilaufgabe_b()
    fig.savefig("3b.pdf")

