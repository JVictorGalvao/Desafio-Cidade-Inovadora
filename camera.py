from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Carregar modelo pré-treinado
# model = YOLO("yolov5s.pt")  # YOLOv5 Small
model = YOLO('yolo11x.pt')  # load a pretrained YOLO detection model

# Caminho para a imagem
image_path = "2.jpg"

# Realizar a detecção (apenas pessoas - classe 0)
results = model(image_path, classes=[0])  # Classe '0' é 'person'

# Obter imagem original
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB

confidence_threshold = 0.01  # Limite de confiabilidade

for box in results[0].boxes:
    confidence = box.conf[0]  # Confiança da detecção
    if confidence >= confidence_threshold:  # Filtrar por confiança
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do bounding box
        label = f"Person: {confidence:.2f}"

        # Desenhar retângulo e rótulo
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)  # Bounding box (verde)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Texto

# Exibir imagem com Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()
