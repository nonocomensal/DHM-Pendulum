import cv2
import numpy as np
import pandas as pd

PEND_VIDEO = 'actuL.mp4'

cor_limite_inferior = np.array([0, 120, 70])
cor_limite_superior = np.array([10, 255, 255])

# Lista para armazenar os dados: [frame, tempo, x, y]
dados = []

# Abre o vídeo
cap = cv2.VideoCapture('actuL.mp4')

# Verifica se o vídeo abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo. Verifique o nome e o caminho do arquivo.")
    exit()

# Obtém o FPS (Frames Por Segundo) do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"O vídeo tem {fps:.2f} FPS.")

if fps > 0:
    delay = int(1000 / fps)
else:
    delay = 1

frame_count = 0
while True:
    # Lê um frame do vídeo
    ret, frame = cap.read()
    
    # Se 'ret' for False, o vídeo acabou
    if not ret:
        break

    # Converte o frame de BGR (padrão do OpenCV) para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Cria uma "máscara" que isola apenas os pixels dentro da faixa de cor definida
    mascara = cv2.inRange(hsv, cor_limite_inferior, cor_limite_superior)
    
    # Melhora a máscara para remover ruídos
    mascara = cv2.erode(mascara, None, iterations=2)
    mascara = cv2.dilate(mascara, None, iterations=2)

    # Encontra os contornos (formas) na máscara
    contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centro = None
    
    # Se algum contorno for encontrado...
    if len(contornos) > 0:
        # Pega o maior contorno
        c = max(contornos, key=cv2.contourArea)
        
        # Encontra o círculo mínimo que envolve o contorno e seu centro
        ((x, y), raio) = cv2.minEnclosingCircle(c)
        
        # Só processa se o raio for de um tamanho mínimo
        if raio > 10:
            # Desenha o círculo e o centro no frame original para visualização
            cv2.circle(frame, (int(x), int(y)), int(raio), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Calcula o tempo em segundos
            tempo = frame_count / fps
            
            # Armazena os dados
            dados.append([frame_count, tempo, x, y])
            print(f"Tempo: {tempo:.2f}s, Posição: (x={x:.1f}, y={y:.1f})")

    # Mostra o vídeo com o rastreamento (pressione 'q' para sair)
    cv2.imshow("Rastreamento do Pendulo", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
        
    frame_count += 1

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()

# --- SALVANDO OS DADOS ---
if dados:
    # Converte a lista de dados para um DataFrame do Pandas
    df = pd.DataFrame(dados, columns=['frame', 'tempo', 'x', 'y'])
    
    # Salva o DataFrame em um arquivo CSV
    df.to_csv('dados_pendulo.csv', index=False, sep=';', decimal=',')

    print("\nDados salvos com sucesso em 'dados_pendulo.csv'")
else:
    print("\nNenhum dado foi coletado. Verifique as configurações de cor e o vídeo.")