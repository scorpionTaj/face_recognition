Ce code implémente un système de reconnaissance faciale pour la gestion de la présence, utilisant Flask pour la partie serveur web, OpenCV pour la détection et la reconnaissance des visages, et scikit-learn pour l'entraînement d'un modèle de reconnaissance faciale. Voici une explication détaillée de chaque section du code :

### Importations
```python
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
```
Les bibliothèques importées servent à :
- `cv2` : Utilisé pour la capture et le traitement d'images (OpenCV).
- `os` : Gérer les opérations du système de fichiers.
- `flask` : Créer l'application web.
- `datetime` : Gérer les dates et heures.
- `numpy` : Manipuler les tableaux numériques.
- `sklearn` : Implémenter le modèle de reconnaissance faciale.
- `pandas` : Manipuler les fichiers CSV.
- `joblib` : Sauvegarder et charger les modèles scikit-learn.

### Initialisation de l'application Flask
```python
app = Flask(__name__)
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```
- `app = Flask(__name__)` : Initialise l'application Flask.
- `nimgs` : Nombre d'images capturées par utilisateur pour l'entraînement.
- `datetoday` et `datetoday2` : Formats de date pour les fichiers de présence.
- `face_detector` : Charge le classifieur de cascade de Haar pour la détection des visages.

### Création des répertoires nécessaires
```python
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static"):
    os.makedirs("static")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")
if f"Attendance-{datetoday}.csv" not in os.listdir("Attendance"):
    with open(f"Attendance/Attendance-{datetoday}.csv", "w") as f:
        f.write("Prènom,ID,Temps")
```
- Vérifie l'existence des répertoires nécessaires et les crée si besoin.
- Crée un fichier CSV pour enregistrer la présence quotidienne si ce fichier n'existe pas.

### Fonctions utilitaires
```python
def totalreg():
    return len(os.listdir("static/faces"))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir("static/faces")
    for user in userlist:
        for imgname in os.listdir(f"static/faces/{user}"):
            img = cv2.imread(f"static/faces/{user}/{imgname}")
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")

def extract_attendance():
    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    names = df["Prènom"]
    rolls = df["ID"]
    times = df["Temps"]
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username, userid = name.split("_")
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    if int(userid) not in list(df["ID"]):
        with open(f"Attendance/Attendance-{datetoday}.csv", "a") as f:
            f.write(f"\n{username},{userid},{current_time}")

def getallusers():
    userlist = os.listdir("static/faces")
    names, rolls = [], []
    for user in userlist:
        name, roll = user.split("_")
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)
```
- `totalreg()` : Retourne le nombre total d'utilisateurs enregistrés.
- `extract_faces(img)` : Extrait les visages d'une image.
- `identify_face(facearray)` : Identifie un visage à partir d'un tableau de caractéristiques.
- `train_model()` : Entraîne le modèle de reconnaissance faciale.
- `extract_attendance()` : Extrait les données de présence depuis le fichier CSV.
- `add_attendance(name)` : Ajoute une entrée de présence pour un utilisateur.
- `getallusers()` : Récupère tous les utilisateurs enregistrés.

### Routes Flask
```python
@app.route("/")
def home():
    names, rolls, times, l = extract_attendance()
    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route("/start", methods=["GET"])
def start():
    names, rolls, times, l = extract_attendance()
    if "face_recognition_model.pkl" not in os.listdir("static"):
        return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess="Il n'y a pas de modèle entraîné dans le dossier statique. Veuillez ajouter un nouveau visage pour continuer.")
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f"{identified_person}", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == ord("o"):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route("/add", methods=["GET", "POST"])
def add():
    newusername = request.form["newusername"]
    newuserid = request.form["newuserid"]
    userimagefolder = "static/faces/" + newusername + "_" + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f"Images Captured: {i}/{nimgs}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + "_" + str(i) + ".jpg"
                cv2.imwrite(userimagefolder + "/" + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow("Adding new User", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
```
- `home()` : Route pour la page d'accueil affichant les informations de présence.
- `start()` : Route pour démarrer la reconnaissance faciale en temps réel et enregistrer la présence.
- `add()` : Route pour ajouter un nouvel utilisateur et capturer des images de son visage.

### Exécution de l'application
```python
if __name__ == "__main__":
    app

.run(debug=True)
```
Démarre l'application Flask en mode debug.

Ce système permet de gérer la présence en détectant et en reconnaissant les visages des utilisateurs via une webcam. Les informations de présence sont stockées dans des fichiers CSV, et le modèle de reconnaissance est entraîné avec les images capturées des visages des utilisateurs.