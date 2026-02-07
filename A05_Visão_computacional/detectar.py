import cv2
from ultralytics import YOLO

# 1. Carregar o Modelo
# 'yolov8n.pt' é a versão "nano" (mais leve/rápida).
# Na primeira vez que você rodar, ele fará o download automático deste arquivo.
print("Carregando modelo...")
model = YOLO('yolov8n.pt')

# 2. Abrir a conexão com a Webcam
# O número '0' geralmente representa a webcam integrada do notebook.
# Se tiver uma USB externa, tente '1'.
endereco_ip = "http://10.106.152.53:8080/video" 

cap = cv2.VideoCapture(endereco_ip)

# Verifica se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

print("Iniciando detecção. Pressione 'q' para sair.")

while True:
    # 3. Ler o frame (imagem) da câmera
    sucesso, frame = cap.read()
    
    if sucesso:
        # 4. Realizar a detecção (Inference)
        # O parâmetro 'conf=0.5' significa que só queremos detecções com 50% ou mais de certeza.
        results = model(frame, conf=0.5)

        # 5. Desenhar as caixas na imagem
        # A função plot() desenha automaticamente os quadrados e nomes na imagem
        annotated_frame = results[0].plot()

        # 6. Mostrar na tela
        cv2.imshow("Visão Computacional - YOLOv8", annotated_frame)

        # 7. Condição de parada (Pressionar 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 8. Limpeza
cap.release()
cv2.destroyAllWindows()