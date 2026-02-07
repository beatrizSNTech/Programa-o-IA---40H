import cv2
from deepface import DeepFace

# Carrega o classificador de rostos padrão do OpenCV (mais leve para detectar ONDE está o rosto)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Iniciando... (A primeira análise pode demorar alguns segundos para carregar o modelo)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Espelhar a imagem (opcional, fica mais natural para webcam)
    frame = cv2.flip(frame, 1) 

    # Converter para tons de cinza (para o detector de rostos simples)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Desenhar retângulo no rosto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Recortar a região do rosto para enviar para a IA (Region of Interest)
        roi_color = frame[y:y+h, x:x+w]

        try:
            # A MÁGICA ACONTECE AQUI
            # O parâmetro 'actions' define o que queremos (emotion, age, gender, race)
            # enforce_detection=False evita que o programa trave se a IA não tiver certeza que é um rosto
            analysis = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            # Pegar a emoção dominante
            # O resultado vem como uma lista, pegamos o primeiro item
            dominant_emotion = analysis[0]['dominant_emotion']
            
            # Traduzir para português (opcional)
            traducoes = {
                'angry': 'Raiva', 'disgust': 'Nojo', 'fear': 'Medo', 
                'happy': 'Feliz', 'sad': 'Triste', 'surprise': 'Surpresa', 
                'neutral': 'Neutro'
            }
            texto = traducoes.get(dominant_emotion, dominant_emotion)

          
            cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        except Exception as e:
          
            pass

    cv2.imshow('Detector de Emoções', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()