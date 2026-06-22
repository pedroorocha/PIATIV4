import cv2
import numpy as np

# =========================
# 1. LEITURA DAS IMAGENS
# =========================
img1 = cv2.imread("gato.jpg", 0)
img2 = cv2.imread("gato1.jpg", 0)

if img1 is None or img2 is None:
    raise ValueError("Imagens não encontradas!")

# garantir binário (0 e 255)
_, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
_, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# =========================
# 2. ELEMENTOS ESTRUTURANTES
# =========================
kernels = {
    "3x3": np.ones((3, 3), np.uint8),
    "5x5": np.ones((5, 5), np.uint8),
    "15x15": np.ones((15, 15), np.uint8),
}

# =========================
# 3. EROSÃO MANUAL
# =========================
def erosion(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    result = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]

            # erosão: todos pixels devem ser 255
            if np.all(region[kernel == 1] == 255):
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result

# =========================
# 4. DILATAÇÃO MANUAL
# =========================
def dilation(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    result = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]

            # dilatação: se algum pixel é 255
            if np.any(region[kernel == 1] == 255):
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result

# =========================
# 5. PROCESSAMENTO COMPLETO
# =========================
def process(img, name):

    for kname, kernel in kernels.items():

        # erosão e dilatação
        eroded = erosion(img, kernel)
        dilated = dilation(img, kernel)

        # abertura = erosão + dilatação
        opening = dilation(eroded, kernel)

        # fechamento = dilatação + erosão
        closing = erosion(dilated, kernel)

        # =========================
        # SALVAMENTO
        # =========================
        cv2.imwrite(f"{name}_erosion_{kname}.png", eroded)
        cv2.imwrite(f"{name}_dilation_{kname}.png", dilated)
        cv2.imwrite(f"{name}_opening_{kname}.png", opening)
        cv2.imwrite(f"{name}_closing_{kname}.png", closing)

# =========================
# 6. EXECUÇÃO NAS 2 IMAGENS
# =========================
process(img1, "img1")
process(img2, "img2")

print("Processamento concluído!")