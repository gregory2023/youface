import cv2
import os

def registrar_usuario(nombre):
    carpeta = f"local_client/usuarios_registrados/{nombre}"
    os.makedirs(carpeta, exist_ok=True)

    # Borrar fotos anteriores si las hay
    for f in os.listdir(carpeta):
        os.remove(os.path.join(carpeta, f))

    cap = cv2.VideoCapture(0)
    fotos_tomadas = 0
    total_fotos = 5

    print(f"[INFO] Vamos a tomar {total_fotos} fotos.")
    print("[INFO] Pulsa ESPACIO para capturar. Cambia ligeramente el angulo entre fotos.")

    while fotos_tomadas < total_fotos:
        ok, frame = cap.read()
        if not ok:
            break

        restantes = total_fotos - fotos_tomadas
        cv2.putText(frame, f"Fotos restantes: {restantes}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Pulsa ESPACIO para capturar", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.imshow("Registro de usuario", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            foto_path = f"{carpeta}/foto_{fotos_tomadas + 1}.jpg"
            cv2.imwrite(foto_path, frame)
            fotos_tomadas += 1
            print(f"[OK] Foto {fotos_tomadas}/{total_fotos} guardada.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Registro completado. {fotos_tomadas} fotos guardadas para {nombre}.")

nombre = input("Introduce tu nombre completo: ")
registrar_usuario(nombre)
