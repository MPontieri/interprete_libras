"""
Detector em Tempo Real para Libras - Versão Rápida
"""
import cv2
import numpy as np
import joblib
from collections import deque
import time
import mediapipe as mp
from scipy import stats
import tensorflow as tf
import pyttsx3
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DetectorLibras:
    def __init__(self):
        self.carregar_modelos()
        self.inicializar_camera()
        self.inicializar_detector()
        
        self.buffer_frames = deque(maxlen=20)
        self.ultima_letra = "?"
        self.confianca = 0
        self.mao_detectada = False
        self.ultimos_landmarks = None
        
        self.fps = 0
        self.fps_count = 0
        self.fps_tempo = time.time()
        
        self.letra_mostrada = "?"
        self.tempo_mostra = 0
        self.ultima_predicao = 0
        self.intervalo_predicao = 0.05  # 50ms entre predicoes
        
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        self.tts.setProperty('volume', 1.0)

        self.ultima_fala = ""
        self.tempo_ultima_fala = 0
        self.intervalo_fala = 2.0  # segundos

    def carregar_modelos(self):
        try:
            self.modelo = tf.keras.models.load_model('modelos/melhor_modelo_libras.keras')
            self.label_encoder = joblib.load('modelos/label_encoder.pkl')
            self.sequencia_tamanho = 20
            print("Modelo carregado")
        except Exception as e:
            print(f"Erro modelo: {e}")
            exit()

    def inicializar_camera(self):
        self.cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            print("Camera nao encontrada")
            exit()

    def inicializar_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path="modelos/hand_landmarker.task")
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("Detector ok")
        except Exception as e:
            print(f"Erro detector: {e}")
            exit()

    def processar_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if detection_result.hand_landmarks:

            if not self.mao_detectada:
                self.buffer_frames.clear()

            self.mao_detectada = True

            # Guarda os landmarks reais para desenhar
            self.ultimos_landmarks = detection_result.hand_landmarks

            left = [0] * 63
            right = [0] * 63

            for hand_landmarks, handedness in zip(
                    detection_result.hand_landmarks,
                    detection_result.handedness):

                label = handedness[0].category_name

                coords = []

                for lm in hand_landmarks:
                    coords.extend([lm.x, lm.y, lm.z])

                if label == "Left":
                    left = coords
                else:
                    right = coords

            # 126 valores (63 esquerda + 63 direita)
            landmarks = left + right

            self.buffer_frames.append(landmarks)

            if len(self.buffer_frames) >= 5:
                self.fazer_predicao_rapida()

        else:
            self.mao_detectada = False
            self.ultimos_landmarks = None
            self.buffer_frames.clear()
            self.ultima_letra = "?"
            self.letra_mostrada = "?"

    def fazer_predicao_rapida(self):
        # Usa os frames disponiveis (repetindo se necessario)
        frames_disponiveis = list(self.buffer_frames)
        
        if len(frames_disponiveis) < self.sequencia_tamanho:
            # Repete o ultimo frame para completar a sequencia
            ultimo_frame = frames_disponiveis[-1]
            frames_faltando = self.sequencia_tamanho - len(frames_disponiveis)
            frames_completos = frames_disponiveis + [ultimo_frame] * frames_faltando
        else:
            frames_completos = frames_disponiveis[-self.sequencia_tamanho:]
        
        sequencia = np.array(frames_completos)
        sequencia = sequencia.reshape(1, self.sequencia_tamanho, -1)
        
        pred_proba = self.modelo.predict(sequencia, verbose=0)[0]
        pred_classe = np.argmax(pred_proba)
        confianca = np.max(pred_proba)
        
        if confianca > 0.75:
            letra = self.label_encoder.inverse_transform([pred_classe])[0]
            self.ultima_letra = letra
            self.confianca = confianca
            self.letra_mostrada = letra
            self.tempo_mostra = time.time()
            agora = time.time()

            if (
                letra != self.ultima_fala
                or agora - self.tempo_ultima_fala >= self.intervalo_fala
            ):

                self.ultima_fala = letra
                self.tempo_ultima_fala = agora

                self.falar_async(letra)

    def desenhar_landmarks(self, frame):

        if self.ultimos_landmarks is None:
            return

        h, w = frame.shape[:2]

        connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12),
            (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20),
            (5,9), (9,13), (13,17)
        ]

        # percorre cada mão detectada
        for hand_landmarks in self.ultimos_landmarks:

            # desenha os pontos
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 3, (0, 200, 0), -1)

            # desenha as conexões
            for idx1, idx2 in connections:

                x1 = int(hand_landmarks[idx1].x * w)
                y1 = int(hand_landmarks[idx1].y * h)

                x2 = int(hand_landmarks[idx2].x * w)
                y2 = int(hand_landmarks[idx2].y * h)

                cv2.line(frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 100, 0),
                        1)
                
    def falar(self, texto):
        try:
            texto = texto.replace("_", " ")
            self.tts.say(texto)
            self.tts.runAndWait()

        except Exception as e:
            print("Erro ao falar:", e)


    def falar_async(self, texto):
        threading.Thread(
            target=self.falar,
            args=(texto,),
            daemon=True
        ).start()
                    

    def desenhar_interface(self, frame):
        h, w = frame.shape[:2]

        overlay = frame.copy()

        # Painel superior
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)

        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # FPS
        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (20, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Status
        status = "MAO DETECTADA" if self.mao_detectada else "AGUARDANDO"

        cor_status = (0, 255, 0) if self.mao_detectada else (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (20, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            cor_status,
            2
        )

        # Buffer
        cv2.putText(
            frame,
            f"BUFFER: {len(self.buffer_frames)}/{self.sequencia_tamanho}",
            (w - 250, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Confiança
        cv2.putText(
            frame,
            f"CONFIANCA: {int(self.confianca*100)}%",
            (w - 250, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Texto reconhecido
        if (
            self.letra_mostrada != "?"
            and time.time() - self.tempo_mostra < 1.5
        ):

            texto = self.letra_mostrada.replace("_", " ")

            if len(texto) <= 4:
                escala = 4
                espessura = 6

            elif len(texto) <= 8:
                escala = 2.8
                espessura = 5

            else:
                escala = 1.8
                espessura = 4

            tamanho = cv2.getTextSize(
                texto,
                cv2.FONT_HERSHEY_DUPLEX,
                escala,
                espessura
            )[0]

            x = (w - tamanho[0]) // 2
            y = h // 2

            # sombra
            cv2.putText(
                frame,
                texto,
                (x + 3, y + 3),
                cv2.FONT_HERSHEY_DUPLEX,
                escala,
                (0, 0, 0),
                espessura + 2
            )

            # texto principal
            cv2.putText(
                frame,
                texto,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                escala,
                (0, 255, 255),
                espessura
            )

            # barra de confiança
            largura = 300
            altura = 25

            bx = (w - largura) // 2
            by = y + 50

            cv2.rectangle(
                frame,
                (bx, by),
                (bx + largura, by + altura),
                (70, 70, 70),
                -1
            )

            cv2.rectangle(
                frame,
                (bx, by),
                (
                    bx + int(largura * self.confianca),
                    by + altura
                ),
                (0, 255, 0),
                -1
            )
    def executar(self):
        print("DETECTOR LIBRAS - MODO RAPIDO")
        print("ESC para sair")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            self.processar_frame(frame)
            
            self.fps_count += 1
            if time.time() - self.fps_tempo >= 1.0:
                self.fps = self.fps_count
                self.fps_count = 0
                self.fps_tempo = time.time()
            
            self.desenhar_landmarks(frame)
            self.desenhar_interface(frame)
            
            cv2.imshow('Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DetectorLibras()
    detector.executar()